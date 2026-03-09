/// Sample from N(mean, std) using Box-Muller transform.
pub fn normal_sample(rng: &mut fastrand::Rng, mean: f32, std: f32) -> f32 {
    let u1 = rng.f64();
    let u2 = rng.f64();
    // Avoid log(0)
    let u1 = if u1 < 1e-30 { 1e-30 } else { u1 };
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (f64::from(mean) + f64::from(std) * z) as f32
}

/// Fill a vector with samples from N(mean, std).
pub fn normal_vec(rng: &mut fastrand::Rng, n: usize, mean: f32, std: f32) -> Vec<f32> {
    (0..n).map(|_| normal_sample(rng, mean, std)).collect()
}
