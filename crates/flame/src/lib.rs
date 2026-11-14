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

#[cfg(not(target_arch = "wasm32"))]
pub mod checkpoint;
pub mod config;
#[cfg(not(target_arch = "wasm32"))]
pub mod control;
#[cfg(not(target_arch = "wasm32"))]
pub mod data;
pub mod estimate;
pub mod generate;
pub mod init;
pub mod model;
pub mod nn;
pub mod optim;
pub mod tokenizer;
#[cfg(not(target_arch = "wasm32"))]
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
        // TF32: matrix multiplications use the tensor cores with a 10-bit mantissa (instead
        // of 23) while storage and accumulation stay fp32. Standard practice for training;
        // 2–8× faster matmuls on Ampere/Hopper data-centre GPUs (A100, H100), smaller gains
        // on consumer cards. Set FLAME_TF32=0 for exact fp32.
        let tf32 = std::env::var("FLAME_TF32").map(|v| v != "0").unwrap_or(true);
        candle_core::cuda::set_gemm_reduced_precision_f32(tf32);
        eprintln!("[flame] running on CUDA GPU 0 (TF32 matmuls {})", if tf32 { "on; FLAME_TF32=0 to disable" } else { "off" });
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
