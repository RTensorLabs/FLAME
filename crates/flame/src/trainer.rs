//! The training loop, shared by every architecture that implements [`LanguageModel`].
//!
//! Features:
//! * gradient accumulation, AdamW, warmup + cosine LR, global-norm gradient clipping
//! * periodic validation, `best/` weights, JSONL metrics
//! * **pause / resume**: Ctrl+C, `--max-minutes`, a `PAUSE` file, and periodic checkpoints
//!   (see [`crate::control`]). Rerunning the same command resumes from the last checkpoint with
//!   identical weights, optimiser moments, LR schedule position and data order.
//! * ETA reporting in hours *and* in "sessions" so you can plan evenings/weekends.

use crate::checkpoint::{self, TrainState};
use crate::config::TrainConfig;
use crate::control::{StopControl, StopReason};
use crate::data::{Batch, BatchSource};
use crate::model::{lm_loss, LanguageModel};
use crate::optim::{grad_norm, lr_at, AdamW};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct TrainOptions {
    /// Stop cleanly after this many minutes of this session.
    pub max_minutes: Option<f64>,
    /// Stop cleanly after this many optimiser steps *in this session*.
    pub session_steps: Option<u64>,
    /// Typical session length in hours, only used to print "≈ N sessions left".
    pub session_hours: f64,
    pub quiet: bool,
}

#[derive(Debug, PartialEq)]
pub enum TrainOutcome {
    Finished { step: u64 },
    Paused { step: u64, reason: String },
}

/// Optional hook to print a text sample during training.
pub type Sampler<'a, M> = &'a dyn Fn(&M) -> Result<String>;

pub struct Trainer<'a, M: LanguageModel> {
    pub model: &'a M,
    pub varmap: &'a mut VarMap,
    pub cfg: &'a TrainConfig,
    pub run_dir: &'a Path,
    pub device: &'a Device,
    pub opts: TrainOptions,
    pub sampler: Option<Sampler<'a, M>>,
}

impl<M: LanguageModel> Trainer<'_, M> {
    pub fn run(&mut self, source: &mut dyn BatchSource) -> Result<TrainOutcome> {
        let cfg = self.cfg;
        std::fs::create_dir_all(self.run_dir)?;
        StopControl::set_quiet(self.opts.quiet);
        let control = StopControl::new(self.run_dir, self.opts.max_minutes);

        let mut opt = AdamW::new(self.varmap, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay)?;
        let mut state = match checkpoint::load(self.run_dir, self.varmap, &mut opt)? {
            Some(s) => {
                self.log(&format!(
                    "resumed from step {} ({:.1} M tokens, {:.2} h trained so far over {} session(s))",
                    s.step,
                    s.tokens_seen as f64 / 1e6,
                    s.train_seconds / 3600.0,
                    s.sessions
                ));
                s
            }
            None => {
                self.log("starting a fresh run");
                TrainState::default()
            }
        };
        state.sessions += 1;
        self.session_log(&format!("session {} started at step {}", state.sessions, state.step))?;

        if state.step >= cfg.max_steps {
            self.log(&format!("run already finished (step {} ≥ max_steps {}).", state.step, cfg.max_steps));
            return Ok(TrainOutcome::Finished { step: state.step });
        }

        self.log(&format!(
            "model: {} | {:.2} M params | {} tokens/step | max_steps {}",
            self.model.describe(),
            self.model.n_params() as f64 / 1e6,
            cfg.tokens_per_step(),
            cfg.max_steps
        ));

        let val = source.val_batches(cfg.eval_batches, self.device)?;
        let vars: Vec<_> = opt.vars().cloned().collect();
        let mut metrics = std::fs::OpenOptions::new().create(true).append(true).open(self.run_dir.join("metrics.jsonl"))?;

        let session_start_step = state.step;
        let mut last_ckpt = Instant::now();
        let mut step_secs_ema: Option<f64> = None;
        let mut eval_secs = 0.0;
        let mut window_start = Instant::now();
        let mut window_tokens = 0u64;
        let mut stop: Option<StopReason> = None;

        while state.step < cfg.max_steps {
            // ---- pause checks (before starting a step) ----
            let eval_due = (state.step + 1) % cfg.eval_every == 0;
            let next = step_secs_ema.unwrap_or(0.0) + if eval_due { eval_secs } else { 0.0 };
            if let Some(r) = control.should_stop(next) {
                stop = Some(r);
                break;
            }
            if let Some(n) = self.opts.session_steps {
                if state.step - session_start_step >= n {
                    stop = Some(StopReason::StepLimit);
                    break;
                }
            }

            // ---- one optimiser step ----
            let t0 = Instant::now();
            let lr = lr_at(state.step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps);
            let mut grads: Vec<Option<Tensor>> = vec![None; vars.len()];
            let mut loss_sum = 0f64;
            let mut stats: Vec<(String, f64)> = vec![];
            let log_now = (state.step + 1) % cfg.log_every == 0;
            for micro in 0..cfg.grad_accum_steps {
                let batch = source.train_batch(state.step, micro, self.device)?;
                self.model.set_mode(true, state.step, micro);
                let out = self.model.forward(&batch.input)?;
                let ce = lm_loss(&out.logits, &batch.target, batch.mask.as_ref())?;
                let loss = match &out.aux_loss {
                    Some(aux) => (&ce + aux)?,
                    None => ce.clone(),
                };
                loss_sum += ce.to_scalar::<f32>()? as f64;
                if log_now && micro == 0 {
                    for (k, v) in &out.stats {
                        stats.push((k.clone(), v.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?));
                    }
                }
                let scaled = (loss / cfg.grad_accum_steps as f64)?;
                let mut store = scaled.backward()?;
                for (i, v) in vars.iter().enumerate() {
                    if let Some(g) = store.remove(v.as_tensor()) {
                        grads[i] = Some(match grads[i].take() {
                            Some(acc) => (acc + g)?,
                            None => g,
                        });
                    }
                }
            }
            let loss = loss_sum / cfg.grad_accum_steps as f64;
            anyhow::ensure!(loss.is_finite(), "loss became {loss} at step {} — lower the learning rate", state.step);
            let gnorm = grad_norm(&grads)?;
            let scale = if cfg.grad_clip > 0.0 && gnorm > cfg.grad_clip { cfg.grad_clip / gnorm } else { 1.0 };
            opt.step(&grads, lr, scale)?;

            state.step += 1;
            state.tokens_seen += cfg.tokens_per_step();
            window_tokens += cfg.tokens_per_step();
            let dt = t0.elapsed().as_secs_f64();
            state.train_seconds += dt;
            step_secs_ema = Some(match step_secs_ema {
                Some(e) => 0.9 * e + 0.1 * dt,
                None => dt,
            });

            // ---- logging ----
            if log_now || state.step == cfg.max_steps {
                let tok_s = window_tokens as f64 / window_start.elapsed().as_secs_f64().max(1e-9);
                window_start = Instant::now();
                window_tokens = 0;
                let remaining_h = (cfg.max_steps - state.step) as f64 * step_secs_ema.unwrap_or(dt) / 3600.0;
                let extra: String = stats.iter().map(|(k, v)| format!(" {k} {v:.3}")).collect();
                self.log(&format!(
                    "step {:>7}/{} | loss {:.4} | lr {:.2e} | gnorm {:.2} | {:.0} tok/s | ETA {}{}",
                    state.step,
                    cfg.max_steps,
                    loss,
                    lr,
                    gnorm,
                    tok_s,
                    format_eta(remaining_h, self.opts.session_hours),
                    extra
                ));
                let stats_json: serde_json::Map<String, serde_json::Value> = stats.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect();
                writeln!(
                    metrics,
                    "{}",
                    serde_json::json!({"step": state.step, "tokens": state.tokens_seen, "loss": loss, "lr": lr,
                        "grad_norm": gnorm, "tok_per_sec": tok_s, "train_hours": state.train_seconds / 3600.0, "stats": stats_json})
                )?;
            }

            // ---- evaluation ----
            if state.step % cfg.eval_every == 0 || state.step == cfg.max_steps {
                let te = Instant::now();
                let vl = evaluate(self.model, &val)?;
                eval_secs = te.elapsed().as_secs_f64();
                let improved = state.best_val_loss.is_none_or(|b| vl < b);
                if improved {
                    state.best_val_loss = Some(vl);
                    checkpoint::save_best(self.run_dir, self.varmap)?;
                }
                self.log(&format!(
                    "  eval @ {} | val_loss {:.4} | ppl {:.2}{}",
                    state.step,
                    vl,
                    vl.exp(),
                    if improved { " | new best ✓" } else { "" }
                ));
                writeln!(metrics, "{}", serde_json::json!({"step": state.step, "tokens": state.tokens_seen, "val_loss": vl}))?;
                if let Some(sampler) = &self.sampler {
                    if cfg.sample_every > 0 && state.step % cfg.sample_every == 0 {
                        self.log(&format!("  sample: {:?}", sampler(self.model)?));
                    }
                }
            }

            // ---- periodic checkpoint (crash / power-loss protection) ----
            if last_ckpt.elapsed().as_secs_f64() >= cfg.checkpoint_every_minutes * 60.0 {
                checkpoint::save(self.run_dir, self.varmap, &opt, &state)?;
                last_ckpt = Instant::now();
                self.log(&format!("  checkpoint saved (step {})", state.step));
            }
        }

        checkpoint::save(self.run_dir, self.varmap, &opt, &state)?;
        match stop {
            Some(reason) if state.step < cfg.max_steps => {
                let msg = format!(
                    "paused at step {}/{} ({}). Checkpoint saved. Run the same command again to resume.",
                    state.step, cfg.max_steps, reason
                );
                self.log(&msg);
                self.session_log(&format!("session {} paused at step {}: {}", state.sessions, state.step, reason))?;
                Ok(TrainOutcome::Paused { step: state.step, reason: reason.to_string() })
            }
            _ => {
                self.log(&format!(
                    "training finished at step {} ({:.2} h of training in total). best val_loss {:.4}",
                    state.step,
                    state.train_seconds / 3600.0,
                    state.best_val_loss.unwrap_or(f64::NAN)
                ));
                self.session_log(&format!("session {} finished the run at step {}", state.sessions, state.step))?;
                Ok(TrainOutcome::Finished { step: state.step })
            }
        }
    }

    fn log(&self, msg: &str) {
        if !self.opts.quiet {
            println!("[flame] {msg}");
        }
    }

    fn session_log(&self, msg: &str) -> Result<()> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.run_dir.join("sessions.log"))?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        writeln!(f, "unix={now} {msg}")?;
        Ok(())
    }
}

/// Mean (masked) cross-entropy over a fixed set of batches.
pub fn evaluate<M: LanguageModel + ?Sized>(model: &M, batches: &[Batch]) -> Result<f64> {
    model.set_mode(false, 0, 0);
    let mut total = 0.0;
    for b in batches {
        let out = model.forward(&b.input)?;
        total += lm_loss(&out.logits, &b.target, b.mask.as_ref())?.to_scalar::<f32>()? as f64;
    }
    Ok(total / batches.len().max(1) as f64)
}

pub fn format_eta(hours: f64, session_hours: f64) -> String {
    let t = if hours < 1.0 { format!("{:.0} min", hours * 60.0) } else { format!("{hours:.1} h") };
    if session_hours > 0.0 && hours > session_hours {
        format!("{t} (≈{:.0} sessions of {session_hours} h)", (hours / session_hours).ceil())
    } else {
        t
    }
}
