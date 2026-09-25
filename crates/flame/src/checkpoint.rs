//! Crash-safe checkpoints.
//!
//! Layout inside a run directory:
//!
//! ```text
//! runs/<name>/
//!   config.toml                 frozen copy of the config the run was started with
//!   checkpoints/latest/         newest resumable state
//!       model.safetensors       weights
//!       optim.safetensors       AdamW moments (m, v)
//!       state.json              step, tokens seen, timing, best val loss
//!   best/model.safetensors      weights with the lowest validation loss so far
//!   metrics.jsonl               one JSON object per log / eval line
//!   sessions.log                when each training session started and why it stopped
//! ```
//!
//! Saving is atomic: everything is written to `latest.tmp/`, then `latest` is renamed to
//! `latest.old`, `latest.tmp` to `latest`, and `latest.old` is deleted. If the laptop dies at
//! any moment, either `latest` or `latest.old` is a complete checkpoint, and [`find_latest`]
//! picks whichever survived.

use crate::optim::AdamW;
use anyhow::{Context, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainState {
    /// Number of completed optimiser steps.
    pub step: u64,
    pub tokens_seen: u64,
    pub best_val_loss: Option<f64>,
    /// Pure training wall-clock seconds summed over all sessions.
    pub train_seconds: f64,
    pub sessions: u64,
}

pub fn ckpt_dir(run_dir: &Path) -> PathBuf {
    run_dir.join("checkpoints")
}

pub fn find_latest(run_dir: &Path) -> Option<PathBuf> {
    let base = ckpt_dir(run_dir);
    ["latest", "latest.old"]
        .iter()
        .map(|n| base.join(n))
        .find(|p| p.join("state.json").exists() && p.join("model.safetensors").exists())
}

pub fn save(run_dir: &Path, varmap: &VarMap, opt: &AdamW, state: &TrainState) -> Result<()> {
    let base = ckpt_dir(run_dir);
    std::fs::create_dir_all(&base)?;
    let tmp = base.join("latest.tmp");
    let latest = base.join("latest");
    let old = base.join("latest.old");
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    varmap.save(tmp.join("model.safetensors"))?;
    opt.save(tmp.join("optim.safetensors"))?;
    // state.json last: its presence marks the directory as complete
    std::fs::write(tmp.join("state.json"), serde_json::to_string_pretty(state)?)?;

    if old.exists() {
        std::fs::remove_dir_all(&old)?;
    }
    if latest.exists() {
        std::fs::rename(&latest, &old)?;
    }
    std::fs::rename(&tmp, &latest)?;
    if old.exists() {
        std::fs::remove_dir_all(&old)?;
    }
    Ok(())
}

/// Loads weights + optimiser + state. Returns `None` if the run has no checkpoint yet.
pub fn load(run_dir: &Path, varmap: &mut VarMap, opt: &mut AdamW) -> Result<Option<TrainState>> {
    let Some(dir) = find_latest(run_dir) else { return Ok(None) };
    let state: TrainState = serde_json::from_str(&std::fs::read_to_string(dir.join("state.json"))?)?;
    varmap.load(dir.join("model.safetensors")).with_context(|| format!("loading weights from {}", dir.display()))?;
    opt.load(dir.join("optim.safetensors"), state.step)?;
    Ok(Some(state))
}

/// Loads only weights: from `best/` if present (and `prefer_best`), else from the latest checkpoint.
pub fn load_weights(run_dir: &Path, varmap: &mut VarMap, prefer_best: bool) -> Result<PathBuf> {
    let best = run_dir.join("best").join("model.safetensors");
    let path = if prefer_best && best.exists() {
        best
    } else if let Some(dir) = find_latest(run_dir) {
        dir.join("model.safetensors")
    } else if best.exists() {
        best
    } else {
        anyhow::bail!("no weights found in {} — train first", run_dir.display());
    };
    varmap.load(&path).with_context(|| format!("loading {}", path.display()))?;
    Ok(path)
}

pub fn save_best(run_dir: &Path, varmap: &VarMap) -> Result<()> {
    let dir = run_dir.join("best");
    std::fs::create_dir_all(&dir)?;
    let tmp = dir.join("model.safetensors.tmp");
    varmap.save(&tmp)?;
    std::fs::rename(tmp, dir.join("model.safetensors"))?;
    Ok(())
}
