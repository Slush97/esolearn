use crate::autograd::backward::Gradients;
use crate::backend::MathBackend;

/// Clip gradients by global L2 norm.
///
/// Computes the total L2 norm across all gradient tensors. If it exceeds
/// `max_norm`, scales all gradients by `max_norm / total_norm`.
///
/// Returns the total norm (before clipping).
pub fn clip_grad_norm<B: MathBackend>(grads: &mut Gradients<B>, max_norm: f32) -> f32 {
    // Compute total L2 norm across all gradients (f64 accumulation)
    let mut total_sq: f64 = 0.0;
    for grad in grads.values() {
        let n = B::norm(grad);
        total_sq += f64::from(n) * f64::from(n);
    }
    let total_norm = total_sq.sqrt() as f32;

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for grad in grads.values_mut() {
            B::scale_inplace(grad, scale);
        }
    }

    total_norm
}

#[cfg(test)]
mod tests {
    use super::clip_grad_norm;
    use crate::autograd::backward::Gradients;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::{DeviceBackend, MathBackend};
    use crate::tensor::shape::Shape;
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    type Cpu = CpuBackend;

    #[test]
    fn clip_scales_when_above_threshold() {
        let mut grads: Gradients<Cpu> = HashMap::new();
        let t = Tensor::<Cpu>::from_vec(vec![0.0; 2], Shape::new(&[2]));
        grads.insert(t.id, Cpu::from_vec(vec![3.0, 4.0], &Shape::new(&[2])));

        let total_norm = clip_grad_norm::<Cpu>(&mut grads, 1.0);
        assert!((total_norm - 5.0).abs() < 1e-5, "total norm should be 5.0");

        let clipped = &grads[&t.id];
        let clipped_norm = Cpu::norm(clipped);
        assert!(
            (clipped_norm - 1.0).abs() < 1e-5,
            "clipped norm should be ~1.0, got {clipped_norm}"
        );
    }

    #[test]
    fn clip_no_change_when_below_threshold() {
        let mut grads: Gradients<Cpu> = HashMap::new();
        let t = Tensor::<Cpu>::from_vec(vec![0.0; 3], Shape::new(&[3]));
        let original = vec![0.1f32, 0.2, 0.3];
        grads.insert(t.id, Cpu::from_vec(original.clone(), &Shape::new(&[3])));

        let total_norm = clip_grad_norm::<Cpu>(&mut grads, 100.0);
        assert!(total_norm < 1.0);

        let result = &grads[&t.id];
        for (a, b) in result.iter().zip(original.iter()) {
            assert!((*a - *b).abs() < 1e-7);
        }
    }

    #[test]
    fn norm_correct_l2() {
        let v = vec![3.0f32, 4.0];
        let n = Cpu::norm(&v);
        assert!((n - 5.0).abs() < 1e-5);
    }
}
