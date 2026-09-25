//! AdamW (Loshchilov & Hutter 2019) implemented by hand.
//!
//! We don't use `candle_nn::AdamW` because its moment buffers are private, and a pausable
//! trainer *must* save them: resuming with fresh (zero) moments causes a loss spike and is
//! not the same run any more.
//!
//! Update for parameter θ with gradient g at step t:
//!   m ← β1·m + (1-β1)·g
//!   v ← β2·v + (1-β2)·g²
//!   m̂ = m / (1-β1^t),  v̂ = v / (1-β2^t)
//!   θ ← θ - lr·wd·θ              (decoupled weight decay; only for matrices, not norm gains)
//!   θ ← θ - lr · m̂ / (sqrt(v̂) + ε)

use anyhow::Result;
use candle_core::{Tensor, Var};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::Path;

pub struct AdamW {
    /// Sorted by name so the order (and thus the checkpoint layout) is stable.
    pub params: Vec<(String, Var)>,
    m: Vec<Var>,
    v: Vec<Var>,
    pub t: u64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamW {
    pub fn new(varmap: &VarMap, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Result<Self> {
        let mut params: Vec<(String, Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let zeros = |v: &Var| Var::zeros(v.shape(), v.dtype(), v.device());
        let m = params.iter().map(|(_, v)| zeros(v)).collect::<candle_core::Result<Vec<_>>>()?;
        let v = params.iter().map(|(_, v)| zeros(v)).collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { params, m, v, t: 0, beta1, beta2, eps, weight_decay })
    }

    pub fn vars(&self) -> impl Iterator<Item = &Var> {
        self.params.iter().map(|(_, v)| v)
    }

    /// Applies one update. `grads[i]` belongs to `self.params[i]` (None = no gradient).
    /// `grad_scale` multiplies every gradient first (used for global-norm clipping).
    pub fn step(&mut self, grads: &[Option<Tensor>], lr: f64, grad_scale: f64) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, (_, theta)) in self.params.iter().enumerate() {
            let Some(g) = &grads[i] else { continue };
            let g = if grad_scale != 1.0 { (g * grad_scale)? } else { g.clone() };
            let m = ((self.m[i].as_tensor() * self.beta1)? + (&g * (1.0 - self.beta1))?)?;
            let v = ((self.v[i].as_tensor() * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?;
            let m_hat = (&m / bc1)?;
            let v_hat = (&v / bc2)?;
            let mut p = theta.as_detached_tensor();
            // Norm gains and other vectors are not decayed (standard practice).
            if self.weight_decay > 0.0 && p.rank() >= 2 {
                p = (p * (1.0 - lr * self.weight_decay))?;
            }
            let update = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
            let p = (p - (update * lr)?)?;
            theta.set(&p.detach())?;
            self.m[i].set(&m.detach())?;
            self.v[i].set(&v.detach())?;
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (i, (name, _)) in self.params.iter().enumerate() {
            map.insert(format!("m.{name}"), self.m[i].as_tensor().clone());
            map.insert(format!("v.{name}"), self.v[i].as_tensor().clone());
        }
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    pub fn load(&mut self, path: impl AsRef<Path>, t: u64) -> Result<()> {
        let device = self.params.first().map(|(_, v)| v.device().clone()).unwrap_or(candle_core::Device::Cpu);
        let map = candle_core::safetensors::load(path, &device)?;
        for (i, (name, _)) in self.params.iter().enumerate() {
            let get = |k: String| map.get(&k).cloned().ok_or_else(|| anyhow::anyhow!("optimizer state missing {k}"));
            self.m[i].set(&get(format!("m.{name}"))?)?;
            self.v[i].set(&get(format!("v.{name}"))?)?;
        }
        self.t = t;
        Ok(())
    }
}

/// Global L2 norm of all gradients.
pub fn grad_norm(grads: &[Option<Tensor>]) -> Result<f64> {
    let mut total = 0f64;
    for g in grads.iter().flatten() {
        total += g.sqr()?.sum_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    }
    Ok(total.sqrt())
}

/// Linear warmup, then cosine decay from `lr` to `min_lr` at `max_steps`.
pub fn lr_at(step: u64, lr: f64, min_lr: f64, warmup: u64, max_steps: u64) -> f64 {
    if step < warmup {
        return lr * (step + 1) as f64 / warmup as f64;
    }
    if step >= max_steps {
        return min_lr;
    }
    let progress = (step - warmup) as f64 / (max_steps - warmup).max(1) as f64;
    min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_shape() {
        assert!((lr_at(0, 1.0, 0.1, 10, 100) - 0.1).abs() < 1e-9);
        assert!((lr_at(9, 1.0, 0.1, 10, 100) - 1.0).abs() < 1e-9);
        assert!((lr_at(10, 1.0, 0.1, 10, 100) - 1.0).abs() < 1e-9);
        assert!((lr_at(100, 1.0, 0.1, 10, 100) - 0.1).abs() < 1e-9);
        let mid = lr_at(55, 1.0, 0.1, 10, 100);
        assert!((mid - 0.55).abs() < 1e-9);
    }
}
