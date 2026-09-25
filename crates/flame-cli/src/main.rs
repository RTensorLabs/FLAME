//! `flame` — command-line interface. Run `flame --help` or `flame <command> --help`.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use flame::checkpoint;
use flame::config::{save_toml, RunConfig};
use flame::data::TokenDataset;
use flame::generate::{generate, SampleOptions};
use flame::model::{LanguageModel, Transformer};
use flame::tokenizer::{BpeTokenizer, Tokenizer};
use flame::trainer::{format_eta, TrainOptions, TrainOutcome, Trainer};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "flame", version, about = "FLAME — Framework for Lightweight AI Model Engineering")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a tokenizer.
    #[command(subcommand)]
    Tokenizer(TokCmd),
    /// Tokenise text files into train.bin / val.bin.
    Prepare {
        /// Files or directories (directories are searched recursively).
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Fraction of documents held out for validation.
        #[arg(long, default_value_t = 0.02)]
        val_fraction: f64,
        /// Document separator inside large files (TinyStories uses <|endoftext|>; for plain
        /// text such as Shakespeare use "\n\n" to treat paragraphs as documents).
        #[arg(long, default_value = "<|endoftext|>")]
        separator: String,
        /// File extensions to include when walking directories.
        #[arg(long, default_value = "txt,md,rs")]
        ext: String,
    },
    /// Train (or resume training) a model. Rerun the same command to resume after a pause.
    Train {
        #[arg(long)]
        config: PathBuf,
        /// Run directory (checkpoints, metrics, logs).
        #[arg(long)]
        run: PathBuf,
        /// Stop cleanly (with a checkpoint) after this many minutes.
        #[arg(long)]
        max_minutes: Option<f64>,
        /// Stop cleanly after this many optimiser steps in this session.
        #[arg(long)]
        steps: Option<u64>,
        /// Length of a typical training session in hours (only for the ETA display).
        #[arg(long, default_value_t = 1.0)]
        session_hours: f64,
        /// Start over: the existing run directory is renamed to <run>.bak-<time>, not deleted.
        #[arg(long)]
        fresh: bool,
        #[arg(long)]
        cpu: bool,
    },
    /// Generate text from a trained run.
    Generate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 200)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Use the latest checkpoint instead of the best one.
        #[arg(long)]
        latest: bool,
        #[arg(long)]
        cpu: bool,
    },
    /// Validation loss / perplexity of a run.
    Eval {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value_t = 50)]
        batches: usize,
        #[arg(long)]
        latest: bool,
        #[arg(long)]
        cpu: bool,
    },
    /// Progress of a run: steps done, hours trained, estimated time left.
    Status {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value_t = 6.0)]
        hours_per_week: f64,
    },
    /// Parameter count of a config.
    Params {
        #[arg(long)]
        config: PathBuf,
    },
    /// Estimate training time for a config on your hardware.
    Estimate {
        #[arg(long)]
        config: PathBuf,
        /// Achieved TFLOP/s. Measure it with a short run; see docs/hardware-and-time-estimates.md.
        #[arg(long, default_value_t = 3.0)]
        tflops: f64,
        #[arg(long, default_value_t = 6.0)]
        hours_per_week: f64,
    },
    /// Write weights + inputs + logits for tools/reference_check.py (PyTorch cross-check).
    ExportReference {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum TokCmd {
    /// Byte-level tokenizer (vocab 258). No training needed.
    Byte {
        #[arg(long)]
        out: PathBuf,
    },
    /// Train a byte-level BPE tokenizer.
    Train {
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        vocab_size: usize,
        #[arg(long)]
        out: PathBuf,
        /// How many MB of text to learn merges from (more = slower, slightly better).
        #[arg(long, default_value_t = 50)]
        sample_mb: usize,
        #[arg(long, default_value = "txt,md,rs")]
        ext: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Tokenizer(TokCmd::Byte { out }) => {
            ensure_parent(&out)?;
            Tokenizer::Byte.save(&out)?;
            println!("wrote byte tokenizer (vocab {}) to {}", Tokenizer::Byte.vocab_size(), out.display());
        }
        Cmd::Tokenizer(TokCmd::Train { input, vocab_size, out, sample_mb, ext }) => {
            let files = collect_files(&input, &ext)?;
            let mut sample = String::new();
            let budget = sample_mb * 1024 * 1024;
            for f in &files {
                if sample.len() >= budget {
                    break;
                }
                let mut buf = Vec::new();
                std::fs::File::open(f)?.take((budget - sample.len()) as u64).read_to_end(&mut buf)?;
                sample.push_str(&String::from_utf8_lossy(&buf).replace(flame::tokenizer::EOS_TEXT, "\n"));
            }
            println!("training BPE (vocab {vocab_size}) on {:.1} MB from {} file(s)…", sample.len() as f64 / 1e6, files.len());
            let t = std::time::Instant::now();
            let tok = Tokenizer::Bpe(BpeTokenizer::train(&sample, vocab_size, true)?);
            ensure_parent(&out)?;
            tok.save(&out)?;
            let probe: String = sample.chars().take(200_000).collect();
            let ratio = probe.len() as f64 / tok.encode(&probe).len() as f64;
            println!("done in {:.0}s → {} (vocab {}, ≈{ratio:.2} bytes/token)", t.elapsed().as_secs_f64(), out.display(), tok.vocab_size());
        }
        Cmd::Prepare { input, tokenizer, out, val_fraction, separator, ext } => {
            // allow writing "\n\n" on the command line
            let separator = separator.replace("\\n", "\n").replace("\\t", "\t");
            prepare(&input, &tokenizer, &out, val_fraction, &separator, &ext)?;
        }
        Cmd::Train { config, run, max_minutes, steps, session_hours, fresh, cpu } => {
            train(&config, &run, max_minutes, steps, session_hours, fresh, cpu)?;
        }
        Cmd::Generate { run, prompt, max_tokens, temperature, top_k, seed, latest, cpu } => {
            let device = flame::select_device(cpu)?;
            let (cfg, tok, model, _vm) = load_run(&run, &device, !latest)?;
            let _ = cfg;
            let ids = generate(
                &model,
                &tok.encode(&prompt),
                &SampleOptions { max_new_tokens: max_tokens, temperature, top_k, seed, stop_token: Some(tok.eos_id()) },
                &device,
            )?;
            println!("{prompt}{}", tok.decode(&ids));
        }
        Cmd::Eval { run, batches, latest, cpu } => {
            let device = flame::select_device(cpu)?;
            let (cfg, _tok, model, _vm) = load_run(&run, &device, !latest)?;
            let mut ds = TokenDataset::open(&cfg.data.dataset_dir, cfg.train.seq_len, cfg.train.micro_batch_size, cfg.train.seed)?;
            let val = flame::data::BatchSource::val_batches(&mut ds, batches, &device)?;
            let loss = flame::trainer::evaluate(&model, &val)?;
            println!("val_loss {loss:.4} | perplexity {:.2} | bits/token {:.3}", loss.exp(), loss / std::f64::consts::LN_2);
        }
        Cmd::Status { run, hours_per_week } => status(&run, hours_per_week)?,
        Cmd::Params { config } => {
            let cfg = config_with_vocab(&config)?;
            let n = Transformer::count_params(&cfg.model);
            let emb = cfg.model.vocab_size * cfg.model.d_model;
            println!("{}", describe_cfg(&cfg));
            println!("total parameters:      {:>12} ({:.2} M)", n, n as f64 / 1e6);
            println!("  embedding:           {:>12}", emb);
            println!("  non-embedding:       {:>12}", n - emb * if cfg.model.tie_embeddings { 1 } else { 2 });
            println!("fp32 weights:          {:>9.1} MB", n as f64 * 4.0 / 1e6);
            println!("training memory (weights+grads+AdamW ≈ 16 B/param): {:.2} GB + activations", n as f64 * 16.0 / 1e9);
        }
        Cmd::Estimate { config, tflops, hours_per_week } => {
            let cfg = config_with_vocab(&config)?;
            let tokens = cfg.train.max_steps as f64 * cfg.train.tokens_per_step() as f64;
            let fwd = Transformer::count_flops_per_token(&cfg.model);
            let attn = flame::estimate::attention_flops_per_token(cfg.model.n_layers, cfg.train.seq_len, cfg.model.d_model);
            println!("{}", describe_cfg(&cfg));
            println!("params {:.2} M | training tokens {:.1} M ({:.0} tokens/param)", Transformer::count_params(&cfg.model) as f64 / 1e6, tokens / 1e6, tokens / Transformer::count_params(&cfg.model) as f64);
            for t in [tflops / 2.0, tflops, tflops * 2.0] {
                let e = flame::estimate::estimate(fwd, attn, tokens, t, hours_per_week);
                println!(
                    "  at {t:>5.1} TFLOP/s achieved: {:>8.1} h  ≈ {:>5.1} weeks at {hours_per_week} h/week  ({:.2e} FLOPs)",
                    e.hours, e.weeks, e.train_flops
                );
            }
            println!("(the middle line is the estimate; measure your real throughput with a 10-minute run and `flame status`)");
        }
        Cmd::ExportReference { config, out } => export_reference(&config, &out)?,
    }
    Ok(())
}

fn ensure_parent(p: &Path) -> Result<()> {
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn collect_files(inputs: &[PathBuf], ext: &str) -> Result<Vec<PathBuf>> {
    let exts: Vec<&str> = ext.split(',').map(|s| s.trim()).collect();
    let mut files = Vec::new();
    for p in inputs {
        if p.is_file() {
            files.push(p.clone());
        } else if p.is_dir() {
            for e in walkdir::WalkDir::new(p).sort_by_file_name().into_iter().filter_map(|e| e.ok()) {
                let path = e.path();
                if path.is_file() && path.extension().and_then(|x| x.to_str()).is_some_and(|x| exts.contains(&x)) {
                    files.push(path.to_path_buf());
                }
            }
        } else {
            anyhow::bail!("input {} does not exist", p.display());
        }
    }
    anyhow::ensure!(!files.is_empty(), "no input files found");
    Ok(files)
}

/// Streams files in ~64 MB blocks, splits them into documents, tokenises documents in parallel,
/// and appends them (each followed by <|endoftext|>) to train.bin or val.bin.
fn prepare(inputs: &[PathBuf], tok_path: &Path, out: &Path, val_fraction: f64, sep: &str, ext: &str) -> Result<()> {
    let tok = Tokenizer::load(tok_path)?;
    anyhow::ensure!(tok.vocab_size() <= 65536, "vocab too large for u16 token files");
    let files = collect_files(inputs, ext)?;
    std::fs::create_dir_all(out)?;
    let val_every = if val_fraction > 0.0 { (1.0 / val_fraction).round().max(2.0) as u64 } else { u64::MAX };
    let mut train_tokens: Vec<u32> = Vec::new();
    let mut val_tokens: Vec<u32> = Vec::new();
    let mut train_w = std::io::BufWriter::new(std::fs::File::create(out.join("train.bin"))?);
    let mut val_w = std::io::BufWriter::new(std::fs::File::create(out.join("val.bin"))?);
    let (mut n_train, mut n_val, mut n_docs, mut n_bytes) = (0u64, 0u64, 0u64, 0u64);
    let eos = tok.eos_id();
    let t0 = std::time::Instant::now();

    let mut flush = |docs: Vec<String>, n_docs: &mut u64| -> Result<()> {
        let encoded: Vec<Vec<u32>> = docs.par_iter().map(|d| tok.encode(d)).collect();
        for ids in encoded {
            if ids.is_empty() {
                continue;
            }
            *n_docs += 1;
            let target = if *n_docs % val_every == 0 { &mut val_tokens } else { &mut train_tokens };
            target.extend_from_slice(&ids);
            target.push(eos);
        }
        n_train += append_tokens(&mut train_w, &mut train_tokens)?;
        n_val += append_tokens(&mut val_w, &mut val_tokens)?;
        Ok(())
    };

    const BLOCK: usize = 64 * 1024 * 1024;
    for f in &files {
        let mut file = std::fs::File::open(f)?;
        let mut carry = String::new();
        loop {
            let mut buf = Vec::with_capacity(BLOCK);
            let n = (&mut file).take(BLOCK as u64).read_to_end(&mut buf)?;
            n_bytes += n as u64;
            let eof = n == 0;
            // don't split a UTF-8 character across blocks
            let valid = match std::str::from_utf8(&buf) {
                Ok(_) => buf.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => buf.len(),
            };
            let rest = buf.split_off(valid);
            carry.push_str(&String::from_utf8_lossy(&buf));
            if !rest.is_empty() {
                // re-read the incomplete character with the next block
                use std::io::{Seek, SeekFrom};
                file.seek(SeekFrom::Current(-(rest.len() as i64)))?;
                n_bytes -= rest.len() as u64;
            }
            let mut docs: Vec<String> = carry.split(sep).map(|s| s.trim_matches('\n').to_string()).collect();
            carry = if eof { String::new() } else { docs.pop().unwrap_or_default() };
            // a file without separators becomes one document per 64 MB block (split at a newline)
            if !eof && docs.is_empty() && carry.len() > BLOCK {
                let cut = carry.rfind('\n').unwrap_or(carry.len());
                docs.push(carry[..cut].to_string());
                carry = carry[cut..].to_string();
            }
            flush(docs, &mut n_docs)?;
            if eof {
                break;
            }
        }
        eprint!("\r  processed {:.1} MB, {} docs", n_bytes as f64 / 1e6, n_docs);
    }
    eprintln!();
    use std::io::Write;
    train_w.flush()?;
    val_w.flush()?;
    let meta = serde_json::json!({
        "tokenizer": tok_path, "vocab_size": tok.vocab_size(), "documents": n_docs,
        "bytes": n_bytes, "train_tokens": n_train, "val_tokens": n_val,
        "bytes_per_token": n_bytes as f64 / (n_train + n_val).max(1) as f64,
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    println!(
        "wrote {} train + {} val tokens ({} docs, {:.2} bytes/token) to {} in {:.0}s",
        n_train,
        n_val,
        n_docs,
        n_bytes as f64 / (n_train + n_val).max(1) as f64,
        out.display(),
        t0.elapsed().as_secs_f64()
    );
    anyhow::ensure!(n_val > 0, "validation split is empty — add more documents or raise --val-fraction");
    Ok(())
}

fn append_tokens(w: &mut impl std::io::Write, tokens: &mut Vec<u32>) -> Result<u64> {
    let n = tokens.len() as u64;
    for t in tokens.drain(..) {
        w.write_all(&(t as u16).to_le_bytes())?;
    }
    Ok(n)
}

fn config_with_vocab(path: &Path) -> Result<RunConfig> {
    let mut cfg = RunConfig::load(path)?;
    if cfg.model.vocab_size == 0 {
        cfg.model.vocab_size = Tokenizer::load(&cfg.data.tokenizer)
            .with_context(|| "config has no vocab_size and the tokenizer could not be loaded (create it first)")?
            .vocab_size();
    }
    Ok(cfg)
}

fn describe_cfg(cfg: &RunConfig) -> String {
    let m = &cfg.model;
    format!(
        "model: d_model {} | layers {} | heads {} | mlp {} | vocab {} | ctx {}",
        m.d_model,
        m.n_layers,
        m.n_heads,
        m.mlp_hidden(),
        m.vocab_size,
        m.max_seq_len
    )
}

fn train(config: &Path, run: &Path, max_minutes: Option<f64>, steps: Option<u64>, session_hours: f64, fresh: bool, cpu: bool) -> Result<()> {
    if fresh && run.exists() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let bak = PathBuf::from(format!("{}.bak-{now}", run.display()));
        std::fs::rename(run, &bak)?;
        println!("[flame] --fresh: moved the old run to {}", bak.display());
    }
    let cfg = config_with_vocab(config)?;
    let frozen = run.join("config.toml");
    if frozen.exists() {
        let old = RunConfig::load(&frozen)?;
        anyhow::ensure!(
            old.model == cfg.model && old.data == cfg.data,
            "the [model]/[data] sections of {} differ from the run's frozen config {}.\n\
             Resuming would load weights into a different model. Use a new --run directory, or --fresh to start over.",
            config.display(),
            frozen.display()
        );
        if old.train != cfg.train {
            println!("[flame] note: [train] settings changed since the run started (e.g. max_steps); using the new values.");
        }
    }
    std::fs::create_dir_all(run)?;
    save_toml(&cfg, &frozen)?;

    let device = flame::select_device(cpu)?;
    let tok = Tokenizer::load(&cfg.data.tokenizer)?;
    anyhow::ensure!(cfg.train.seq_len <= cfg.model.max_seq_len, "train.seq_len must be ≤ model.max_seq_len");
    let (mut varmap, model) = flame::build_transformer(&cfg.model, &device, cfg.train.seed)?;
    let mut data = TokenDataset::open(&cfg.data.dataset_dir, cfg.train.seq_len, cfg.train.micro_batch_size, cfg.train.seed)?;
    println!(
        "[flame] dataset: {:.1} M train tokens → this run sees {:.1} M tokens ({:.1} epochs)",
        data.train.len() as f64 / 1e6,
        (cfg.train.max_steps * cfg.train.tokens_per_step()) as f64 / 1e6,
        (cfg.train.max_steps * cfg.train.tokens_per_step()) as f64 / data.train.len() as f64
    );
    let dev2 = device.clone();
    let tok2 = tok.clone();
    let sampler = move |m: &Transformer| -> Result<String> {
        let ids = generate(m, &[], &SampleOptions { max_new_tokens: 60, stop_token: Some(tok2.eos_id()), ..Default::default() }, &dev2)?;
        Ok(tok2.decode(&ids))
    };
    let mut trainer = Trainer {
        model: &model,
        varmap: &mut varmap,
        cfg: &cfg.train,
        run_dir: run,
        device: &device,
        opts: TrainOptions { max_minutes, session_steps: steps, session_hours, quiet: false },
        sampler: Some(&sampler),
    };
    match trainer.run(&mut data)? {
        TrainOutcome::Finished { .. } => println!("[flame] done. Try: flame generate --run {}", run.display()),
        TrainOutcome::Paused { .. } => {}
    }
    Ok(())
}

fn load_run(run: &Path, device: &Device, prefer_best: bool) -> Result<(RunConfig, Tokenizer, Transformer, candle_nn::VarMap)> {
    let cfg = RunConfig::load(run.join("config.toml")).with_context(|| format!("{} is not a FLAME run directory", run.display()))?;
    let tok = Tokenizer::load(&cfg.data.tokenizer)?;
    let (mut vm, model) = flame::build_transformer(&cfg.model, device, cfg.train.seed)?;
    let path = checkpoint::load_weights(run, &mut vm, prefer_best)?;
    eprintln!("[flame] loaded {}", path.display());
    Ok((cfg, tok, model, vm))
}

fn status(run: &Path, hours_per_week: f64) -> Result<()> {
    let cfg = RunConfig::load(run.join("config.toml"))?;
    let Some(dir) = checkpoint::find_latest(run) else {
        println!("no checkpoint yet in {}", run.display());
        return Ok(());
    };
    let st: checkpoint::TrainState = serde_json::from_str(&std::fs::read_to_string(dir.join("state.json"))?)?;
    let done = st.step as f64 / cfg.train.max_steps as f64;
    println!("run:            {}", run.display());
    println!("progress:       step {} / {} ({:.1}%)", st.step, cfg.train.max_steps, 100.0 * done);
    println!("tokens seen:    {:.1} M", st.tokens_seen as f64 / 1e6);
    println!("time trained:   {:.2} h over {} session(s)", st.train_seconds / 3600.0, st.sessions);
    if let Some(b) = st.best_val_loss {
        println!("best val loss:  {b:.4} (ppl {:.2})", b.exp());
    }
    if st.step > 0 && st.step < cfg.train.max_steps {
        let left_h = st.train_seconds / st.step as f64 * (cfg.train.max_steps - st.step) as f64 / 3600.0;
        println!("time left:      {} ≈ {:.1} weeks at {hours_per_week} h/week", format_eta(left_h, 1.0), left_h / hours_per_week);
        let tok_s = st.tokens_seen as f64 / st.train_seconds.max(1e-9);
        let fl = Transformer::count_flops_per_token(&cfg.model) + flame::estimate::attention_flops_per_token(cfg.model.n_layers, cfg.train.seq_len, cfg.model.d_model);
        println!("throughput:     {:.0} tokens/s ≈ {:.2} TFLOP/s achieved (use this with `flame estimate --tflops`)", tok_s, 3.0 * fl * tok_s / 1e12);
    } else if st.step >= cfg.train.max_steps {
        println!("status:         finished");
    }
    Ok(())
}

fn export_reference(config: &Path, out: &Path) -> Result<()> {
    let cfg = config_with_vocab(config)?;
    let device = Device::Cpu;
    let (vm, model) = flame::build_transformer(&cfg.model, &device, cfg.train.seed)?;
    std::fs::create_dir_all(out)?;
    vm.save(out.join("model.safetensors"))?;
    let t = cfg.model.max_seq_len.min(32);
    let ids: Vec<u32> = (0..2 * t).map(|i| ((i * 31 + 7) % cfg.model.vocab_size) as u32).collect();
    let input = Tensor::from_vec(ids, (2, t), &device)?;
    let logits = model.forward(&input)?.logits;
    let mut io: HashMap<String, Tensor> = HashMap::new();
    io.insert("input".into(), input.to_dtype(candle_core::DType::I64)?);
    io.insert("logits".into(), logits);
    candle_core::safetensors::save(&io, out.join("io.safetensors"))?;
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg.model)?)?;
    println!("wrote reference bundle to {} — now run: python tools/reference_check.py {}", out.display(), out.display());
    Ok(())
}
