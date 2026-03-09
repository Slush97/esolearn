/// Cosine decay with linear warmup — industry standard for GPT-2 training.
pub struct CosineScheduler {
    warmup_steps: usize,
    total_steps: usize,
    peak_lr: f32,
    min_lr: f32,
}

impl CosineScheduler {
    pub fn new(warmup_steps: usize, total_steps: usize, peak_lr: f32, min_lr: f32) -> Self {
        assert!(
            warmup_steps <= total_steps,
            "warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
        );
        assert!(
            peak_lr >= min_lr,
            "peak_lr ({peak_lr}) must be >= min_lr ({min_lr})"
        );
        Self {
            warmup_steps,
            total_steps,
            peak_lr,
            min_lr,
        }
    }

    /// Get the learning rate for a given step.
    ///
    /// - Warmup: linear ramp `0 -> peak_lr` over `warmup_steps`
    /// - Decay: cosine anneal `peak_lr -> min_lr` over remaining steps
    /// - After `total_steps`: clamp to `min_lr`
    pub fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.min_lr;
        }

        if step < self.warmup_steps {
            // Linear warmup: 0 -> peak_lr
            return self.peak_lr * (step as f32 / self.warmup_steps as f32);
        }

        // Cosine decay: peak_lr -> min_lr
        let decay_steps = self.total_steps - self.warmup_steps;
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.peak_lr - self.min_lr) * cosine as f32
    }
}
