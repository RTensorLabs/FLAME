//! Back-of-the-envelope training-time estimates (see docs/hardware-and-time-estimates.md).
//!
//! Training compute ≈ 6 · N · D FLOPs (N = parameters touched per token, D = tokens):
//! 2·N for the forward pass and ≈ 4·N for the backward pass. Attention-score FLOPs
//! (≈ 12 · layers · seq_len · d_model per token) are added separately because they matter
//! for small models with long contexts.

#[derive(Debug, Clone)]
pub struct Estimate {
    pub train_flops: f64,
    pub hours: f64,
    pub weeks: f64,
}

/// * `flops_per_token_fwd`: forward FLOPs per token (≈ 2 · params)
/// * `effective_tflops`: *achieved* throughput, not the GPU's marketing number
/// * `hours_per_week`: how much time you actually train each week
pub fn estimate(flops_per_token_fwd: f64, attn_flops_per_token_fwd: f64, tokens: f64, effective_tflops: f64, hours_per_week: f64) -> Estimate {
    let train_flops = 3.0 * (flops_per_token_fwd + attn_flops_per_token_fwd) * tokens;
    let hours = train_flops / (effective_tflops * 1e12) / 3600.0;
    Estimate { train_flops, hours, weeks: hours / hours_per_week.max(1e-9) }
}

/// Forward attention-score FLOPs per token for a causal model (QKᵀ and PV, averaged over positions).
pub fn attention_flops_per_token(n_layers: usize, seq_len: usize, d_model: usize) -> f64 {
    // 2 matmuls × 2 FLOPs/MAC × (T/2 average causal length) × d_model per layer
    2.0 * 2.0 * (seq_len as f64 / 2.0) * d_model as f64 * n_layers as f64
}
