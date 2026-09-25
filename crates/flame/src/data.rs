//! Datasets and batching.
//!
//! Text is tokenised **once** by `flame prepare` into flat binary files of little-endian `u16`
//! token ids (`train.bin`, `val.bin`), the same format nanoGPT uses. Training memory-maps these
//! files, so a multi-GB dataset costs almost no RAM.
//!
//! Batches are sampled at random offsets from an RNG seeded by `(seed, step, micro_step)`.
//! This makes the data order a pure function of the step number, so **resuming after a pause
//! reproduces exactly the batches an uninterrupted run would have seen** — no data-loader state
//! has to be saved.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One (micro-)batch. `input[b, t]` predicts `target[b, t]`.
pub struct Batch {
    /// (B, T) u32
    pub input: Tensor,
    /// (B, T) u32
    pub target: Tensor,
    /// Optional (B, T) f32 weights; 0 = ignore this position in the loss (padding, prompts).
    pub mask: Option<Tensor>,
}

/// Anything that can produce training and validation batches.
pub trait BatchSource {
    /// Must be deterministic in `(step, micro)` so that resumed runs see the same data.
    fn train_batch(&mut self, step: u64, micro: usize, device: &Device) -> Result<Batch>;
    /// A fixed, deterministic validation set.
    fn val_batches(&mut self, n: usize, device: &Device) -> Result<Vec<Batch>>;
}

/// Mixes a seed with the step so every (step, micro) pair gets an independent RNG stream.
pub fn step_rng(seed: u64, step: u64, micro: usize) -> StdRng {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    for v in [step, micro as u64] {
        // splitmix64
        x = x.wrapping_add(v.wrapping_mul(0xBF58_476D_1CE4_E5B9)).wrapping_add(0x94D0_49BB_1331_11EB);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
    }
    StdRng::seed_from_u64(x)
}

/// A memory-mapped file of u16 token ids.
pub struct TokenFile {
    mmap: Mmap,
}

impl TokenFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let f = File::open(path).with_context(|| format!("opening {} (did you run `flame prepare`?)", path.display()))?;
        // SAFETY: the file is only read, and FLAME never writes to a .bin while training.
        let mmap = unsafe { Mmap::map(&f)? };
        anyhow::ensure!(mmap.len() % 2 == 0, "{} is not a u16 token file", path.display());
        Ok(Self { mmap })
    }

    pub fn len(&self) -> usize {
        self.mmap.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        u16::from_le_bytes([self.mmap[2 * i], self.mmap[2 * i + 1]]) as u32
    }

    pub fn slice(&self, start: usize, len: usize) -> Vec<u32> {
        (start..start + len).map(|i| self.get(i)).collect()
    }
}

#[allow(dead_code)]
pub fn write_token_file(path: impl AsRef<Path>, tokens: impl IntoIterator<Item = u32>) -> Result<u64> {
    let mut w = BufWriter::new(File::create(path)?);
    let mut n = 0u64;
    for t in tokens {
        anyhow::ensure!(t <= u16::MAX as u32, "token id {t} does not fit in u16");
        w.write_all(&(t as u16).to_le_bytes())?;
        n += 1;
    }
    w.flush()?;
    Ok(n)
}

/// Random contiguous windows from `train.bin` / `val.bin` (next-token prediction).
pub struct TokenDataset {
    pub train: TokenFile,
    pub val: TokenFile,
    pub seq_len: usize,
    pub batch_size: usize,
    pub seed: u64,
}

impl TokenDataset {
    pub fn open(dir: impl AsRef<Path>, seq_len: usize, batch_size: usize, seed: u64) -> Result<Self> {
        let dir = dir.as_ref();
        let train = TokenFile::open(dir.join("train.bin"))?;
        let val = TokenFile::open(dir.join("val.bin"))?;
        for (name, f) in [("train", &train), ("val", &val)] {
            anyhow::ensure!(
                f.len() > seq_len + 1,
                "{name}.bin has only {} tokens; need more than seq_len+1 = {}",
                f.len(),
                seq_len + 1
            );
        }
        Ok(Self { train, val, seq_len, batch_size, seed })
    }

    fn batch_from(&self, file: &TokenFile, rng: &mut StdRng, device: &Device) -> Result<Batch> {
        let (b, t) = (self.batch_size, self.seq_len);
        let mut inp = Vec::with_capacity(b * t);
        let mut tgt = Vec::with_capacity(b * t);
        for _ in 0..b {
            let start = rng.random_range(0..file.len() - t - 1);
            let window = file.slice(start, t + 1);
            inp.extend_from_slice(&window[..t]);
            tgt.extend_from_slice(&window[1..]);
        }
        Ok(Batch {
            input: Tensor::from_vec(inp, (b, t), device)?,
            target: Tensor::from_vec(tgt, (b, t), device)?,
            mask: None,
        })
    }
}

impl BatchSource for TokenDataset {
    fn train_batch(&mut self, step: u64, micro: usize, device: &Device) -> Result<Batch> {
        let mut rng = step_rng(self.seed, step, micro);
        self.batch_from(&self.train, &mut rng, device)
    }

    fn val_batches(&mut self, n: usize, device: &Device) -> Result<Vec<Batch>> {
        // Fixed seed, independent of the training seed: val loss is comparable across runs.
        let mut rng = StdRng::seed_from_u64(0xFEED);
        (0..n).map(|_| self.batch_from(&self.val, &mut rng, device)).collect()
    }
}

/// In-memory dataset: used by tests and by the "overfit a tiny dataset" sanity check.
pub struct InMemoryDataset {
    pub tokens: Vec<u32>,
    pub seq_len: usize,
    pub batch_size: usize,
    pub seed: u64,
}

impl InMemoryDataset {
    fn batch(&self, rng: &mut StdRng, device: &Device) -> Result<Batch> {
        let (b, t) = (self.batch_size, self.seq_len);
        let mut inp = Vec::with_capacity(b * t);
        let mut tgt = Vec::with_capacity(b * t);
        for _ in 0..b {
            let start = rng.random_range(0..self.tokens.len() - t);
            inp.extend_from_slice(&self.tokens[start..start + t]);
            tgt.extend_from_slice(&self.tokens[start + 1..start + t + 1]);
        }
        Ok(Batch {
            input: Tensor::from_vec(inp, (b, t), device)?,
            target: Tensor::from_vec(tgt, (b, t), device)?,
            mask: None,
        })
    }
}

impl BatchSource for InMemoryDataset {
    fn train_batch(&mut self, step: u64, micro: usize, device: &Device) -> Result<Batch> {
        self.batch(&mut step_rng(self.seed, step, micro), device)
    }
    fn val_batches(&mut self, n: usize, device: &Device) -> Result<Vec<Batch>> {
        let mut rng = StdRng::seed_from_u64(0xFEED);
        (0..n).map(|_| self.batch(&mut rng, device)).collect()
    }
}
