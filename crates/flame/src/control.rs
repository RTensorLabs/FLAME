//! Pause control: the ways a training session can be asked to stop *cleanly*
//! (save a checkpoint, then exit so the next run resumes exactly where it left off).
//!
//! 1. **Ctrl+C** (or the OS sending SIGTERM/SIGHUP at shutdown/log-out): finish the current
//!    step, checkpoint, exit. Press Ctrl+C a second time to quit immediately *without* saving
//!    (you then lose at most `checkpoint_every_minutes` of work).
//! 2. **Time budget** `--max-minutes N`: stop before the budget runs out. Ideal for the
//!    "I have 45 minutes tonight" sessions.
//! 3. **PAUSE file**: create an empty file called `PAUSE` in the run directory (from another
//!    terminal, a file manager, anything). The trainer notices within one step, saves, deletes
//!    the file and exits.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Once;
use std::time::Instant;

static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);
static HANDLER: Once = Once::new();
static QUIET: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    Interrupted,
    TimeBudget,
    PauseFile,
    StepLimit,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Interrupted => write!(f, "Ctrl+C / shutdown signal"),
            StopReason::TimeBudget => write!(f, "time budget (--max-minutes) reached"),
            StopReason::PauseFile => write!(f, "PAUSE file found"),
            StopReason::StepLimit => write!(f, "session step limit (--steps) reached"),
        }
    }
}

pub struct StopControl {
    started: Instant,
    budget_secs: Option<f64>,
    pause_file: PathBuf,
}

impl StopControl {
    pub fn new(run_dir: &Path, max_minutes: Option<f64>) -> Self {
        HANDLER.call_once(|| {
            let res = ctrlc::set_handler(|| {
                let n = INTERRUPTS.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    if !QUIET.load(Ordering::SeqCst) {
                        eprintln!("\n[flame] stop requested — finishing this step and saving a checkpoint… (press Ctrl+C again to quit WITHOUT saving)");
                    }
                } else {
                    eprintln!("\n[flame] second Ctrl+C: exiting immediately without saving.");
                    std::process::exit(130);
                }
            });
            if let Err(e) = res {
                eprintln!("[flame] warning: could not install Ctrl+C handler: {e}");
            }
        });
        Self { started: Instant::now(), budget_secs: max_minutes.map(|m| m * 60.0), pause_file: run_dir.join("PAUSE") }
    }

    pub fn set_quiet(quiet: bool) {
        QUIET.store(quiet, Ordering::SeqCst);
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// `next_step_secs`: expected duration of the next step (+ eval if due), so we stop
    /// *before* overrunning the budget rather than after.
    pub fn should_stop(&self, next_step_secs: f64) -> Option<StopReason> {
        if INTERRUPTS.load(Ordering::SeqCst) > 0 {
            return Some(StopReason::Interrupted);
        }
        if self.pause_file.exists() {
            let _ = std::fs::remove_file(&self.pause_file);
            return Some(StopReason::PauseFile);
        }
        if let Some(budget) = self.budget_secs {
            // keep ~30 s (or 2 steps) of headroom for saving the checkpoint
            let reserve = (2.0 * next_step_secs).max(30.0).min(budget * 0.1);
            if self.elapsed_secs() + next_step_secs + reserve >= budget {
                return Some(StopReason::TimeBudget);
            }
        }
        None
    }

    /// Clears a previous interrupt (used by tests that run several sessions in one process).
    pub fn reset_interrupts() {
        INTERRUPTS.store(0, Ordering::SeqCst);
    }

    /// Simulates Ctrl+C (tests).
    pub fn request_stop() {
        INTERRUPTS.fetch_add(1, Ordering::SeqCst);
    }
}
