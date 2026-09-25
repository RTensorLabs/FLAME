//! Run configuration, loaded from TOML files in `configs/`.
//!
//! A run config has three sections:
//!
//! ```toml
//! [model]   # architecture hyper-parameters (see ModelConfig)
//! [train]   # optimisation + pause/checkpoint settings (see TrainConfig)
//! [data]    # where the tokenised data lives (see DataConfig)
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Hyper-parameters of the decoder-only Transformer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Filled in from the tokenizer at runtime when set to 0 / omitted.
    #[serde(default)]
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Hidden size of the SwiGLU MLP. 0 = LLaMA default (≈ 8/3 · d_model, rounded up to 64).
    #[serde(default)]
    pub mlp_hidden: usize,
    pub max_seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Share the token-embedding matrix with the output (LM head) projection.
    /// Saves `vocab_size · d_model` parameters, which matters a lot for tiny models.
    #[serde(default = "default_true")]
    pub tie_embeddings: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn mlp_hidden(&self) -> usize {
        if self.mlp_hidden > 0 {
            self.mlp_hidden
        } else {
            llama_mlp_hidden(self.d_model)
        }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.vocab_size > 0, "vocab_size must be set (it is normally taken from the tokenizer)");
        anyhow::ensure!(self.d_model.is_multiple_of(self.n_heads), "d_model must be divisible by n_heads");
        anyhow::ensure!(self.head_dim().is_multiple_of(2), "head_dim must be even for RoPE");
        Ok(())
    }
}

/// LLaMA's MLP sizing rule: 2/3 of 4·d, rounded up to a multiple of 64.
pub fn llama_mlp_hidden(d_model: usize) -> usize {
    let h = (8 * d_model).div_ceil(3);
    h.div_ceil(64) * 64
}

/// Optimisation, evaluation and checkpoint/pause settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Sequences per forward/backward pass. Lower this if you run out of VRAM.
    pub micro_batch_size: usize,
    /// Micro-batches accumulated per optimiser step.
    /// Effective batch = micro_batch_size · grad_accum_steps · seq_len tokens.
    #[serde(default = "default_one")]
    pub grad_accum_steps: usize,
    pub seq_len: usize,
    pub max_steps: u64,
    pub lr: f64,
    #[serde(default)]
    pub min_lr: f64,
    #[serde(default)]
    pub warmup_steps: u64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// Clip the global gradient L2 norm to this value (0 = off).
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f64,
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
    #[serde(default = "default_eval_batches")]
    pub eval_batches: usize,
    #[serde(default = "default_log_every")]
    pub log_every: u64,
    /// Save a checkpoint at least this often (wall-clock minutes). Protects against crashes,
    /// power loss and forgotten laptops going to sleep.
    #[serde(default = "default_ckpt_minutes")]
    pub checkpoint_every_minutes: f64,
    /// Print a text sample every N steps (0 = never). Only used when a tokenizer is available.
    #[serde(default)]
    pub sample_every: u64,
}

impl TrainConfig {
    pub fn tokens_per_step(&self) -> u64 {
        (self.micro_batch_size * self.grad_accum_steps * self.seq_len) as u64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Tokenizer JSON produced by `flame tokenizer ...`.
    pub tokenizer: String,
    /// Directory produced by `flame prepare` (contains train.bin / val.bin).
    pub dataset_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub model: ModelConfig,
    pub train: TrainConfig,
    pub data: DataConfig,
}

impl RunConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_toml(path)
    }
}

pub fn load_toml<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn save_toml<T: Serialize>(value: &T, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path.as_ref(), toml::to_string_pretty(value)?)?;
    Ok(())
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_seed() -> u64 {
    1337
}
fn default_one() -> usize {
    1
}
fn default_weight_decay() -> f64 {
    0.1
}
fn default_beta1() -> f64 {
    0.9
}
fn default_beta2() -> f64 {
    0.95
}
fn default_eps() -> f64 {
    1e-8
}
fn default_grad_clip() -> f64 {
    1.0
}
fn default_eval_every() -> u64 {
    250
}
fn default_eval_batches() -> usize {
    20
}
fn default_log_every() -> u64 {
    10
}
fn default_ckpt_minutes() -> f64 {
    10.0
}
