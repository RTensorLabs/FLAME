//! Autoregressive sampling.
//!
//! Deliberately simple: the whole context is re-run for every new token (no KV cache). For the
//! tiny models in Phases 1–2 this is fast enough and keeps the model code free of caching
//! logic. A KV cache is a good first "runtime" exercise later (see docs/roadmap.md).

use crate::model::LanguageModel;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct SampleOptions {
    pub max_new_tokens: usize,
    /// 0 = greedy (always the most likely token).
    pub temperature: f64,
    /// Only sample among the k most likely tokens (0 = all).
    pub top_k: usize,
    pub seed: u64,
    /// Stop when this token is produced.
    pub stop_token: Option<u32>,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self { max_new_tokens: 200, temperature: 0.8, top_k: 40, seed: 42, stop_token: None }
    }
}

pub fn generate<M: LanguageModel + ?Sized>(model: &M, prompt: &[u32], opts: &SampleOptions, device: &Device) -> Result<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut ids = prompt.to_vec();
    if ids.is_empty() {
        ids.push(crate::tokenizer::EOS_ID); // start from a document boundary
    }
    let mut out = Vec::new();
    for _ in 0..opts.max_new_tokens {
        let ctx_start = ids.len().saturating_sub(model.max_seq_len());
        let ctx = &ids[ctx_start..];
        let input = Tensor::new(ctx, device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?.logits;
        let last = logits.i((0, ctx.len() - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let next = sample(&last, opts.temperature, opts.top_k, &mut rng);
        if Some(next) == opts.stop_token {
            break;
        }
        ids.push(next);
        out.push(next);
    }
    Ok(out)
}

pub fn sample(logits: &[f32], temperature: f64, top_k: usize, rng: &mut StdRng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    if top_k > 0 && top_k < logits.len() {
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(top_k);
    }
    let max = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = idx.iter().map(|&i| (((logits[i] - max) as f64) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.random::<f64>() * total;
    for (w, &i) in weights.iter().zip(&idx) {
        r -= w;
        if r <= 0.0 {
            return i as u32;
        }
    }
    *idx.last().unwrap() as u32
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0)
}
