//! Models. [`LanguageModel`] is the interface the trainer, evaluator and generator use, so
//! experimental architectures (see the `research/` crate) plug into the same infrastructure.

use crate::config::ModelConfig;
use crate::nn::{causal_mask, Block, Rope, RmsNorm};
use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, Init, VarBuilder};

pub struct ModelOutput {
    /// (B, T, V) next-token logits.
    pub logits: Tensor,
    /// Extra loss terms the model wants optimised (e.g. router load-balancing). Scalar.
    pub aux_loss: Option<Tensor>,
    /// Named scalar diagnostics (e.g. average routing steps). Only read on log steps.
    pub stats: Vec<(String, Tensor)>,
}

pub trait LanguageModel {
    /// input: (B, T) u32 token ids.
    fn forward(&self, input: &Tensor) -> Result<ModelOutput>;
    fn max_seq_len(&self) -> usize;
    /// Total trainable parameters.
    fn n_params(&self) -> usize;
    /// Approximate forward FLOPs per token (2 · multiply-adds), excluding attention scores.
    fn flops_per_token(&self) -> f64;
    fn describe(&self) -> String;
    /// Called by the trainer before every micro-batch (`training = true`) and before
    /// evaluation (`false`). Models with training-only behaviour (e.g. routing exploration)
    /// derive their randomness from `(step, micro)` so resumed runs stay deterministic.
    fn set_mode(&self, _training: bool, _step: u64, _micro: usize) {}
}

/// Decoder-only Transformer in the LLaMA style:
/// token embedding → N × [RMSNorm → causal MHA with RoPE → residual → RMSNorm → SwiGLU → residual]
/// → RMSNorm → LM head.
pub struct Transformer {
    tok_emb: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Option<candle_nn::Linear>,
    rope: Rope,
    cfg: ModelConfig,
}

impl Transformer {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate().map_err(candle_core::Error::msg)?;
        let emb_w = vb.pp("tok_emb").get_with_hints((cfg.vocab_size, cfg.d_model), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?;
        let tok_emb = Embedding::new(emb_w, cfg.d_model);
        let blocks = (0..cfg.n_layers)
            .map(|i| Block::new(cfg.d_model, cfg.n_heads, cfg.mlp_hidden(), cfg.norm_eps, cfg.n_layers, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.d_model, cfg.norm_eps, vb.pp("norm"))?;
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(crate::nn::linear(cfg.d_model, cfg.vocab_size, 0.02, vb.pp("lm_head"))?)
        };
        let rope = Rope::new(cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta, vb.device())?;
        Ok(Self { tok_emb, blocks, norm, lm_head, rope, cfg: cfg.clone() })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Parameter count computed from the config alone (used by `flame params` / `estimate`).
    pub fn count_params(cfg: &ModelConfig) -> usize {
        let emb = cfg.vocab_size * cfg.d_model;
        let head = if cfg.tie_embeddings { 0 } else { emb };
        emb + head + cfg.n_layers * Block::n_params(cfg.d_model, cfg.mlp_hidden()) + cfg.d_model
    }

    /// Forward FLOPs per token ≈ 2 · (non-embedding params + LM head).
    pub fn count_flops_per_token(cfg: &ModelConfig) -> f64 {
        let body = cfg.n_layers * Block::n_params(cfg.d_model, cfg.mlp_hidden());
        2.0 * (body + cfg.vocab_size * cfg.d_model) as f64
    }
}

impl LanguageModel for Transformer {
    fn forward(&self, input: &Tensor) -> Result<ModelOutput> {
        let (_, t) = input.dims2()?;
        if t > self.cfg.max_seq_len {
            candle_core::bail!("sequence length {t} exceeds max_seq_len {}", self.cfg.max_seq_len);
        }
        let mask = causal_mask(t, input.device())?;
        let mut x = self.tok_emb.forward(input)?;
        for block in &self.blocks {
            x = block.forward(&x, &self.rope, &mask)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&x)?,
            None => x.broadcast_matmul(&self.tok_emb.embeddings().t()?)?,
        };
        Ok(ModelOutput { logits, aux_loss: None, stats: vec![] })
    }

    fn max_seq_len(&self) -> usize {
        self.cfg.max_seq_len
    }

    fn n_params(&self) -> usize {
        Self::count_params(&self.cfg)
    }

    fn flops_per_token(&self) -> f64 {
        Self::count_flops_per_token(&self.cfg)
    }

    fn describe(&self) -> String {
        format!(
            "Transformer(d_model={}, layers={}, heads={}, mlp_hidden={}, vocab={}, ctx={}, tied={})",
            self.cfg.d_model,
            self.cfg.n_layers,
            self.cfg.n_heads,
            self.cfg.mlp_hidden(),
            self.cfg.vocab_size,
            self.cfg.max_seq_len,
            self.cfg.tie_embeddings
        )
    }
}

/// Cross-entropy over (B, T, V) logits, optionally weighted by a (B, T) mask.
pub fn lm_loss(logits: &Tensor, target: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits = logits.reshape((b * t, v))?;
    let target = target.reshape(b * t)?;
    match mask {
        None => candle_nn::loss::cross_entropy(&logits, &target),
        Some(mask) => {
            let logp = candle_nn::ops::log_softmax(&logits, 1)?;
            let nll = logp.gather(&target.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?; // (B·T)
            let mask = mask.reshape(b * t)?;
            let denom = mask.sum_all()?.maximum(1.0)?;
            (nll * &mask)?.sum_all()?.div(&denom)
        }
    }
}
