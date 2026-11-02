//! Binary batch bridge for the optional accelerated trainer. Uses the exact
//! native FLAME mixer/RNG so checkpoint step numbers retain their data order.
use anyhow::{bail, Result};
use candle_core::Device;
use flame::{config::RunConfig, data::{Batch, BatchSource, MixedDataset}};
use std::io::{self, BufRead, BufWriter, Write};

fn write_batch(batch: &Batch, out: &mut impl Write) -> Result<()> {
    for tensor in [&batch.input, &batch.target] {
        for id in tensor.flatten_all()?.to_vec1::<u32>()? {
            out.write_all(&id.to_le_bytes())?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let config = std::env::args().nth(1).expect("batch-server <config.toml>");
    let cfg = RunConfig::load(config)?;
    let mut ds = MixedDataset::open(&cfg.data.resolved_sources()?, cfg.train.seq_len,
                                    cfg.train.micro_batch_size, cfg.train.seed)?;
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("t") => {
                let step: u64 = parts.next().expect("step").parse()?;
                for micro in 0..cfg.train.grad_accum_steps {
                    write_batch(&ds.train_batch(step, micro, &Device::Cpu)?, &mut out)?;
                }
            }
            Some("v") => {
                let n: usize = parts.next().expect("number of batches").parse()?;
                for batch in ds.val_batches(n, &Device::Cpu)? {
                    write_batch(&batch, &mut out)?;
                }
            }
            _ => bail!("expected t <step> or v <batches>"),
        }
        out.flush()?;
    }
    Ok(())
}
