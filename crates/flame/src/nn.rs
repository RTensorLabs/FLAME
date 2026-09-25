//! Transformer building blocks, written out explicitly (no fused kernels) so every step is
//! readable and differentiable with Candle's autograd.
//!
//! Tensor shape conventions: B = batch, T = sequence length, D = d_model,
//! H = number of heads, Hd = head dim, V = vocab size.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Init, Linear, VarBuilder};

/// Linear layer (no bias) initialised N(0, std²) like GPT-2 / LLaMA.
pub fn linear(in_dim: usize, out_dim: usize, std: f64, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", Init::Randn { mean: 0.0, stdev: std })?;
    Ok(Linear::new(w, None))
}

/// SiLU(x) = x · σ(x), written with primitive ops.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    x / (x.neg()?.exp()? + 1.0)?
}

/// Root-mean-square LayerNorm (Zhang & Sennrich 2019), as used by LLaMA:
///   y = x / sqrt(mean(x²) + eps) · g
/// Unlike LayerNorm there is no mean-centering and no bias.
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(ms + self.eps)?.sqrt()?)?;
        x.broadcast_mul(&self.weight)
    }
}

/// Rotary position embeddings (Su et al. 2021), "rotate-half" layout as in LLaMA / HF.
///
/// Each pair of channels (i, i + Hd/2) is rotated by angle `pos · θ^(-2i/Hd)`, so the dot
/// product q·k depends only on the *relative* distance between positions.
pub struct Rope {
    cos: Tensor, // (max_T, Hd)
    sin: Tensor, // (max_T, Hd)
}

impl Rope {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half).map(|i| (theta.powf(-((2 * i) as f64) / head_dim as f64)) as f32).collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let pos = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?.reshape((max_seq_len, 1))?;
        let freqs = pos.broadcast_mul(&inv_freq)?; // (T, Hd/2)
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // (T, Hd)
        Ok(Self { cos: emb.cos()?, sin: emb.sin()? })
    }

    /// x: (B, H, T, Hd)
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, t, hd) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;
        let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
        let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

/// (T, T) additive mask: 0 on/below the diagonal, a large negative number above it, so a
/// position can never attend to the future.
pub fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t).flat_map(|i| (0..t).map(move |j| if j > i { -1e9 } else { 0.0 })).collect();
    Tensor::from_vec(mask, (t, t), device)
}

/// Multi-head causal self-attention with RoPE.
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    /// `out_std` is the init std of the output projection (scaled down for deep residual stacks).
    pub fn new(d_model: usize, n_heads: usize, std: f64, out_std: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear(d_model, d_model, std, vb.pp("q_proj"))?,
            k_proj: linear(d_model, d_model, std, vb.pp("k_proj"))?,
            v_proj: linear(d_model, d_model, std, vb.pp("v_proj"))?,
            o_proj: linear(d_model, d_model, out_std, vb.pp("o_proj"))?,
            n_heads,
            head_dim: d_model / n_heads,
        })
    }

    /// x: (B, T, D), mask: (T, T) → (B, T, D)
    pub fn forward(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let split = |p: &Linear| -> Result<Tensor> {
            p.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = rope.apply(&split(&self.q_proj)?)?;
        let k = rope.apply(&split(&self.k_proj)?)?;
        let v = split(&self.v_proj)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?; // (B, H, T, T)
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let y = probs.matmul(&v)?; // (B, H, T, Hd)
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        self.o_proj.forward(&y)
    }

    pub fn n_params(d_model: usize) -> usize {
        4 * d_model * d_model
    }
}

/// SwiGLU feed-forward block (Shazeer 2020), as in LLaMA:
///   y = W2 · (SiLU(W1 x) ⊙ W3 x)
pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    pub fn new(d_model: usize, hidden: usize, std: f64, out_std: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear(d_model, hidden, std, vb.pp("gate"))?,
            up: linear(d_model, hidden, std, vb.pp("up"))?,
            down: linear(hidden, d_model, out_std, vb.pp("down"))?,
        })
    }

    pub fn n_params(d_model: usize, hidden: usize) -> usize {
        3 * d_model * hidden
    }
}

impl Module for SwiGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = silu(&self.gate.forward(x)?)?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

/// Pre-norm Transformer block:
///   x = x + Attn(RMSNorm(x))
///   x = x + MLP(RMSNorm(x))
pub struct Block {
    attn_norm: RmsNorm,
    attn: CausalSelfAttention,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

impl Block {
    pub fn new(d_model: usize, n_heads: usize, mlp_hidden: usize, eps: f64, n_layers: usize, vb: VarBuilder) -> Result<Self> {
        let std = 0.02;
        // GPT-2 trick: shrink the init of projections that write into the residual stream so
        // its variance does not grow with depth.
        let out_std = 0.02 / (2.0 * n_layers as f64).sqrt();
        Ok(Self {
            attn_norm: RmsNorm::new(d_model, eps, vb.pp("attn_norm"))?,
            attn: CausalSelfAttention::new(d_model, n_heads, std, out_std, vb.pp("attn"))?,
            mlp_norm: RmsNorm::new(d_model, eps, vb.pp("mlp_norm"))?,
            mlp: SwiGlu::new(d_model, mlp_hidden, std, out_std, vb.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?, rope, mask)?)?;
        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
    }

    pub fn n_params(d_model: usize, mlp_hidden: usize) -> usize {
        CausalSelfAttention::n_params(d_model) + SwiGlu::n_params(d_model, mlp_hidden) + 2 * d_model
    }
}
