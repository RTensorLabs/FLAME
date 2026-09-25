//! End-to-end checks of the Phase 1/2 requirements:
//! * the model is causal (future tokens cannot influence past logits),
//! * the model can overfit a tiny dataset (the "if it can't overfit, don't scale" gate),
//! * pause + resume produces *bit-identical* weights to an uninterrupted run.

use candle_core::{Device, IndexOp, Tensor};
use flame::config::{ModelConfig, TrainConfig};
use flame::control::StopControl;
use flame::data::InMemoryDataset;
use flame::model::LanguageModel;
use flame::trainer::{TrainOptions, TrainOutcome, Trainer};
use flame::{build_transformer, tokenizer::Tokenizer};
use std::sync::Mutex;

// The Ctrl+C flag is process-global, so tests that train must not overlap.
static SERIAL: Mutex<()> = Mutex::new(());

fn tiny_model(vocab: usize) -> ModelConfig {
    ModelConfig { vocab_size: vocab, d_model: 64, n_layers: 2, n_heads: 4, mlp_hidden: 0, max_seq_len: 64, rope_theta: 10_000.0, norm_eps: 1e-5, tie_embeddings: true }
}

fn tiny_train(max_steps: u64) -> TrainConfig {
    TrainConfig {
        seed: 7, micro_batch_size: 8, grad_accum_steps: 2, seq_len: 32, max_steps, lr: 3e-3, min_lr: 3e-4, warmup_steps: 10,
        weight_decay: 0.0, beta1: 0.9, beta2: 0.99, eps: 1e-8, grad_clip: 1.0, eval_every: 50, eval_batches: 2,
        log_every: 50, checkpoint_every_minutes: 100.0, sample_every: 0, init_from: None,
    }
}

fn tmp_run(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("flame_test_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn model_is_causal() {
    let dev = Device::Cpu;
    let (_vm, model) = build_transformer(&tiny_model(258), &dev, 1).unwrap();
    let a: Vec<u32> = (0..16).map(|i| (i * 7 % 250) as u32).collect();
    let mut b = a.clone();
    for x in b.iter_mut().skip(10) {
        *x = 3; // change only the future
    }
    let la = model.forward(&Tensor::new(a.as_slice(), &dev).unwrap().unsqueeze(0).unwrap()).unwrap().logits;
    let lb = model.forward(&Tensor::new(b.as_slice(), &dev).unwrap().unsqueeze(0).unwrap()).unwrap().logits;
    let diff_past = (la.i((.., ..10)).unwrap() - lb.i((.., ..10)).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let diff_future = (la.i((.., 10..)).unwrap() - lb.i((.., 10..)).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff_past < 1e-5, "future tokens leaked into the past: {diff_past}");
    assert!(diff_future > 1e-4, "changing inputs should change later logits");
}

#[test]
fn overfits_tiny_dataset() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    StopControl::reset_interrupts();
    let dev = Device::Cpu;
    let text = "FLAME learns to repeat this exact sentence. Rust ownership is simple once it clicks. ";
    let tok = Tokenizer::Byte;
    let tokens = tok.encode(&text.repeat(4));
    let mcfg = tiny_model(tok.vocab_size());
    let tcfg = tiny_train(300);
    let (mut varmap, model) = build_transformer(&mcfg, &dev, 1).unwrap();
    let mut data = InMemoryDataset { tokens, seq_len: tcfg.seq_len, batch_size: tcfg.micro_batch_size, seed: 1 };
    let run = tmp_run("overfit");
    let mut tr = Trainer { model: &model, varmap: &mut varmap, cfg: &tcfg, run_dir: &run, device: &dev, opts: TrainOptions { quiet: true, ..Default::default() }, sampler: None };
    assert_eq!(tr.run(&mut data).unwrap(), TrainOutcome::Finished { step: 300 });
    let val = flame::data::BatchSource::val_batches(&mut data, 4, &dev).unwrap();
    let loss = flame::trainer::evaluate(&model, &val).unwrap();
    assert!(loss < 0.15, "a tiny model must be able to memorise a tiny dataset; loss = {loss}");
}

#[test]
fn pause_and_resume_is_bit_identical() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    StopControl::reset_interrupts();
    let dev = Device::Cpu;
    let tok = Tokenizer::Byte;
    let tokens = tok.encode(&"the quick brown fox jumps over the lazy dog. ".repeat(20));
    let mcfg = tiny_model(tok.vocab_size());
    let tcfg = tiny_train(30);
    let opts = TrainOptions { quiet: true, ..Default::default() };
    let data = || InMemoryDataset { tokens: tokens.clone(), seq_len: 32, batch_size: 8, seed: 3 };

    // A: uninterrupted
    let run_a = tmp_run("resume_a");
    let (mut vm_a, model_a) = build_transformer(&mcfg, &dev, 5).unwrap();
    Trainer { model: &model_a, varmap: &mut vm_a, cfg: &tcfg, run_dir: &run_a, device: &dev, opts: opts.clone(), sampler: None }.run(&mut data()).unwrap();

    // B: three sessions (12 + 11 + rest), each in a fresh model object, as after a reboot
    let run_b = tmp_run("resume_b");
    for (i, n) in [Some(12), Some(11), None].into_iter().enumerate() {
        let (mut vm_b, model_b) = build_transformer(&mcfg, &dev, 5 + i as u64 * 100).unwrap(); // different init: must be overwritten by the checkpoint (except in session 0)
        if i == 0 {
            let (v, m) = build_transformer(&mcfg, &dev, 5).unwrap();
            vm_b = v;
            let o = TrainOptions { session_steps: n, ..opts.clone() };
            let out = Trainer { model: &m, varmap: &mut vm_b, cfg: &tcfg, run_dir: &run_b, device: &dev, opts: o, sampler: None }.run(&mut data()).unwrap();
            assert!(matches!(out, TrainOutcome::Paused { step: 12, .. }));
            continue;
        }
        let o = TrainOptions { session_steps: n, ..opts.clone() };
        Trainer { model: &model_b, varmap: &mut vm_b, cfg: &tcfg, run_dir: &run_b, device: &dev, opts: o, sampler: None }.run(&mut data()).unwrap();
    }

    let load = |run: &std::path::Path| candle_core::safetensors::load(run.join("checkpoints/latest/model.safetensors"), &dev).unwrap();
    let (wa, wb) = (load(&run_a), load(&run_b));
    assert_eq!(wa.len(), wb.len());
    for (k, ta) in &wa {
        let d = (ta - &wb[k]).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0, "weight {k} differs after resume by {d}");
    }
}

#[test]
fn ctrl_c_pauses_and_saves() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::Cpu;
    let tok = Tokenizer::Byte;
    let mcfg = tiny_model(tok.vocab_size());
    let tcfg = tiny_train(1000);
    let (mut vm, model) = build_transformer(&mcfg, &dev, 1).unwrap();
    let mut data = InMemoryDataset { tokens: tok.encode(&"abc ".repeat(100)), seq_len: 32, batch_size: 2, seed: 1 };
    let run = tmp_run("ctrlc");
    StopControl::reset_interrupts();
    StopControl::request_stop(); // as if the user pressed Ctrl+C before the first step
    let out = Trainer { model: &model, varmap: &mut vm, cfg: &tcfg, run_dir: &run, device: &dev, opts: TrainOptions { quiet: true, ..Default::default() }, sampler: None }.run(&mut data).unwrap();
    StopControl::reset_interrupts();
    assert!(matches!(out, TrainOutcome::Paused { step: 0, .. }));
    assert!(flame::checkpoint::find_latest(&run).is_some());
    // PAUSE file variant
    std::fs::write(run.join("PAUSE"), "").unwrap();
    let out = Trainer { model: &model, varmap: &mut vm, cfg: &tcfg, run_dir: &run, device: &dev, opts: TrainOptions { quiet: true, ..Default::default() }, sampler: None }.run(&mut data).unwrap();
    assert!(matches!(out, TrainOutcome::Paused { .. }));
    assert!(!run.join("PAUSE").exists(), "PAUSE file should be consumed");
}
