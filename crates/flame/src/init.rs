//! Seeded, device-independent weight initialisation.
//!
//! Candle cannot seed its CPU random generator, and GPU RNG streams differ from CPU ones. To
//! make runs reproducible (and CPU/GPU initialisations identical) FLAME draws every initial
//! weight from its own seeded RNG on the host, then uploads it to the device.

use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Mutex;

pub struct SeededBackend {
    varmap: VarMap,
    rng: Mutex<StdRng>,
}

impl SeededBackend {
    pub fn new(varmap: VarMap, seed: u64) -> Self {
        Self { varmap, rng: Mutex::new(StdRng::seed_from_u64(seed)) }
    }
}

/// A VarBuilder whose new variables are stored in `varmap` and initialised from `seed`.
pub fn seeded_var_builder(varmap: &VarMap, seed: u64, device: &Device) -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(SeededBackend::new(varmap.clone(), seed)), DType::F32, device.clone())
}

fn normal(rng: &mut StdRng) -> f64 {
    // Box–Muller transform
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

impl SimpleBackend for SeededBackend {
    fn get(&self, s: Shape, name: &str, h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let mut data = self.varmap.data().lock().unwrap();
        if let Some(v) = data.get(name) {
            if v.shape() != &s {
                candle_core::bail!("shape mismatch for {name}: {:?} vs {:?}", v.shape(), s);
            }
            return Ok(v.as_tensor().clone());
        }
        let n = s.elem_count();
        let values: Vec<f32> = {
            let mut rng = self.rng.lock().unwrap();
            match h {
                Init::Const(c) => vec![c as f32; n],
                Init::Randn { mean, stdev } => (0..n).map(|_| (mean + stdev * normal(&mut rng)) as f32).collect(),
                Init::Uniform { lo, up } => (0..n).map(|_| rng.random_range(lo..up) as f32).collect(),
                Init::Kaiming { .. } => candle_core::bail!("{name}: use an explicit Randn init in FLAME models"),
            }
        };
        let t = Tensor::from_vec(values, s, &Device::Cpu)?.to_dtype(dtype)?.to_device(dev)?;
        let var = Var::from_tensor(&t)?;
        let out = var.as_tensor().clone();
        data.insert(name.to_string(), var);
        Ok(out)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        candle_core::bail!("{name}: get_unchecked is not supported")
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.varmap.data().lock().unwrap().contains_key(name)
    }
}
