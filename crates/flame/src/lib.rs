//! # FLAME — Framework for Lightweight AI Model Engineering
//!
//! Small, readable infrastructure for training language models from scratch with
//! [Candle](https://github.com/huggingface/candle):
//!
//! | module        | what it contains                                                  |
//! |---------------|-------------------------------------------------------------------|
//! | [`tokenizer`] | byte-level and BPE tokenizers (training, encode, decode)          |
//! | [`data`]      | memory-mapped token files, deterministic resumable batching       |
//! | [`nn`]        | RMSNorm, RoPE, causal self-attention, SwiGLU, Transformer block   |
//! | [`model`]     | `LanguageModel` trait, decoder-only `Transformer`, LM loss         |
//! | [`optim`]     | hand-written AdamW with saveable state, LR schedule, grad clipping |
//! | [`trainer`]   | pausable/resumable training loop with evaluation and ETA          |
//! | [`control`]   | Ctrl+C / time-budget / PAUSE-file stop handling                    |
//! | [`checkpoint`]| atomic, crash-safe checkpoints                                    |
//! | [`generate`]  | temperature / top-k sampling                                      |
//! | [`estimate`]  | training-time estimates                                           |

pub mod checkpoint;
pub mod config;
pub mod control;
pub mod data;
pub mod estimate;
pub mod generate;
pub mod init;
pub mod model;
pub mod nn;
pub mod optim;
pub mod tokenizer;
pub mod trainer;

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarMap;

/// Picks the CUDA GPU if FLAME was built with `--features cuda` and a GPU is present,
/// otherwise the CPU. `force_cpu` always selects the CPU.
pub fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    let d = Device::cuda_if_available(0)?;
    if d.is_cpu() {
        eprintln!("[flame] running on CPU (build with `--features cuda` to use an NVIDIA GPU)");
    } else {
        eprintln!("[flame] running on CUDA GPU 0");
    }
    Ok(d)
}

/// Creates a fresh, randomly initialised Transformer and the VarMap holding its weights.
/// Initialisation is seeded so that runs are reproducible.
pub fn build_transformer(cfg: &config::ModelConfig, device: &Device, seed: u64) -> Result<(VarMap, model::Transformer)> {
    let varmap = VarMap::new();
    let vb = init::seeded_var_builder(&varmap, seed, device);
    let model = model::Transformer::new(cfg, vb)?;
    Ok((varmap, model))
}
