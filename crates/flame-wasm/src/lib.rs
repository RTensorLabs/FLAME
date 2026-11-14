//! FLAME in the browser.
//!
//! JavaScript downloads the run's `config.toml`, the tokenizer JSON and the `model.safetensors`
//! bytes, builds a [`Flame`] from them, then calls [`Flame::step`] once per token so the page
//! can show text as it is written. Sampling is the same code the `flame generate` CLI uses
//! ([`flame::generate::next_token`]), so the same seed gives the same text.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use flame::config::ModelConfig;
use flame::generate::next_token;
use flame::model::{LanguageModel, Transformer};
use flame::tokenizer::{Tokenizer, EOS_ID};
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Only the `[model]` table of a run's config.toml is needed for generation.
#[derive(Deserialize)]
struct RunFile {
    model: ModelConfig,
}

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct Flame {
    model: Transformer,
    tok: Tokenizer,
    device: Device,
    /// Prompt tokens followed by everything generated so far.
    ids: Vec<u32>,
    /// Index in `ids` where generated tokens start.
    start: usize,
    rng: StdRng,
    temperature: f64,
    top_k: usize,
    done: bool,
}

#[wasm_bindgen]
impl Flame {
    /// `weights` may be stored as f32, f16 or bf16; they are converted to f32 on load.
    #[wasm_bindgen(constructor)]
    pub fn new(config_toml: &str, tokenizer_json: &str, weights: &[u8]) -> Result<Flame, JsError> {
        console_error_panic_hook::set_once();
        let device = Device::Cpu;
        let cfg = toml::from_str::<RunFile>(config_toml).map_err(js_err)?.model;
        let tok = Tokenizer::from_json(tokenizer_json).map_err(js_err)?;
        let tensors = candle_core::safetensors::load_buffer(weights, &device)
            .map_err(js_err)?
            .into_iter()
            .map(|(name, t)| Ok((name, t.to_dtype(DType::F32)?)))
            .collect::<candle_core::Result<_>>()
            .map_err(js_err)?;
        let model = Transformer::new(&cfg, VarBuilder::from_tensors(tensors, DType::F32, &device)).map_err(js_err)?;
        Ok(Flame {
            model,
            tok,
            device,
            ids: Vec::new(),
            start: 0,
            rng: StdRng::seed_from_u64(0),
            temperature: 0.8,
            top_k: 40,
            done: true,
        })
    }

    /// Short description, e.g. for the page header.
    pub fn describe(&self) -> String {
        format!("{} ({:.1}M parameters)", self.model.describe(), self.model.n_params() as f64 / 1e6)
    }

    /// Context length in tokens.
    #[wasm_bindgen(js_name = maxSeqLen)]
    pub fn max_seq_len(&self) -> usize {
        self.model.max_seq_len()
    }

    /// Number of tokens `text` encodes to.
    #[wasm_bindgen(js_name = countTokens)]
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tok.encode(text).len()
    }

    /// Begins a new generation. Same meaning as the CLI flags; `seed` is a u32 so JavaScript
    /// can pass a plain number.
    pub fn start(&mut self, prompt: &str, temperature: f64, top_k: usize, seed: u32) {
        self.ids = self.tok.encode(prompt);
        if self.ids.is_empty() {
            self.ids.push(EOS_ID); // start from a document boundary, like the CLI
        }
        self.start = self.ids.len();
        self.rng = StdRng::seed_from_u64(seed as u64);
        self.temperature = temperature;
        self.top_k = top_k;
        self.done = false;
    }

    /// Generates one token. Returns all text generated so far (decoded together, so multi-byte
    /// characters split across tokens come out right), or `undefined` once the model has
    /// produced end-of-text.
    pub fn step(&mut self) -> Result<Option<String>, JsError> {
        if self.done {
            return Ok(None);
        }
        let next = next_token(&self.model, &self.ids, self.temperature, self.top_k, &mut self.rng, &self.device).map_err(js_err)?;
        if next == self.tok.eos_id() {
            self.done = true;
            return Ok(None);
        }
        self.ids.push(next);
        Ok(Some(self.tok.decode(&self.ids[self.start..])))
    }

    /// Tokens generated so far.
    #[wasm_bindgen(js_name = generatedTokens)]
    pub fn generated_tokens(&self) -> usize {
        self.ids.len() - self.start
    }
}
