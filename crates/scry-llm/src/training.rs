use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::autograd::backward::{backward, Gradients};
use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::data::{Batch, DataLoader};
use crate::nn::gpt2::{Gpt2Config, Gpt2Model};
use crate::nn::Module;
use crate::optim::adamw::{AdamW, AdamWConfig};
use crate::optim::clip::clip_grad_norm;
use crate::optim::scheduler::CosineScheduler;
use crate::tensor::TensorId;

/// Training configuration.
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub peak_lr: f32,
    pub min_lr: f32,
    pub grad_accum_steps: usize,
    pub max_grad_norm: f32,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: PathBuf,
    pub seed: u64,
    /// Use gradient checkpointing to trade compute for memory.
    pub use_checkpointing: bool,
    /// How many transformer blocks per checkpoint segment.
    pub checkpoint_every: usize,
    /// Peak GPU TFLOPS for MFU reporting (e.g. 23.1 for FP32, 88 for BF16 tensor cores).
    /// If `None`, MFU is omitted from log output.
    pub peak_tflops: Option<f64>,
    /// Total trainable parameters (set by caller for MFU computation).
    pub n_params: Option<usize>,
}

/// Metrics from a single training step.
pub struct StepMetrics {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f32,
    pub tokens_per_sec: f64,
    pub mfu: f64,
}

/// Training loop orchestrator.
pub struct Trainer<B: MathBackend> {
    pub model: Gpt2Model<B>,
    pub optimizer: AdamW<B>,
    pub scheduler: CosineScheduler,
    pub config: TrainingConfig,
    pub model_config: Gpt2Config,
    pub step: usize,
    pub rng: fastrand::Rng,
    /// Cached set of parameter IDs exempt from weight decay.
    pub no_decay: HashSet<TensorId>,
}

impl<B: MathBackend> Trainer<B> {
    pub fn new(model: Gpt2Model<B>, model_config: Gpt2Config, config: TrainingConfig) -> Self {
        let scheduler = CosineScheduler::new(
            config.warmup_steps,
            config.total_steps,
            config.peak_lr,
            config.min_lr,
        );
        let optimizer = AdamW::new(AdamWConfig {
            lr: config.peak_lr,
            ..AdamWConfig::default()
        });
        let rng = fastrand::Rng::with_seed(config.seed);
        let no_decay = model.no_decay_ids();

        Self {
            model,
            optimizer,
            scheduler,
            config,
            model_config,
            step: 0,
            rng,
            no_decay,
        }
    }

    /// Load a trainer from a checkpoint.
    #[cfg(feature = "safetensors")]
    pub fn from_checkpoint(
        path: &std::path::Path,
        model_config: Gpt2Config,
        config: TrainingConfig,
    ) -> crate::error::Result<Self> {
        let (model, optimizer, step, seed) =
            crate::checkpoint::load_checkpoint::<B>(path, &model_config)?;
        let scheduler = CosineScheduler::new(
            config.warmup_steps,
            config.total_steps,
            config.peak_lr,
            config.min_lr,
        );
        let rng = fastrand::Rng::with_seed(seed);
        let no_decay = model.no_decay_ids();
        Ok(Self {
            model,
            optimizer,
            scheduler,
            config,
            model_config,
            step,
            rng,
            no_decay,
        })
    }

    /// Run one training step with gradient accumulation over micro-batches.
    ///
    /// `batches` should have `grad_accum_steps` entries (or fewer if at end of data).
    pub fn train_step(&mut self, batches: &[Batch]) -> StepMetrics {
        let lr = self.scheduler.get_lr(self.step);
        self.optimizer.set_lr(lr);

        let n_accum = batches.len();
        let mut accumulated_grads: Gradients<B> = HashMap::new();
        let mut total_loss = 0.0f32;

        for batch in batches {
            let mut tape = GradTape::<B>::new();
            let logits = if self.config.use_checkpointing {
                self.model.forward_batch_checkpointed(
                    &batch.input_ids,
                    batch.batch_size,
                    batch.seq_len,
                    self.config.checkpoint_every,
                    &mut self.rng,
                    &mut tape,
                )
            } else {
                self.model.forward_batch(
                    &batch.input_ids,
                    batch.batch_size,
                    batch.seq_len,
                    &mut self.rng,
                    &mut tape,
                )
            };
            let loss = ops::cross_entropy(
                &logits,
                &batch.targets,
                batch.batch_size * batch.seq_len,
                self.model_config.vocab_size,
                Some(&mut tape),
            );
            total_loss += loss.to_vec()[0];

            let grads = if self.config.use_checkpointing {
                self.model.backward_checkpointed(&tape, loss.id)
            } else {
                backward(&tape, loss.id)
            };
            merge_grads::<B>(&mut accumulated_grads, grads);
        }

        // NaN/Inf safety net: skip optimizer step if loss is bad
        let avg_loss = total_loss / n_accum as f32;
        if avg_loss.is_nan() || avg_loss.is_infinite() {
            eprintln!(
                "step {:>6} | WARNING: loss is {avg_loss}, skipping optimizer step",
                self.step + 1
            );
            self.step += 1;
            return StepMetrics {
                loss: avg_loss,
                grad_norm: 0.0,
                lr,
                tokens_per_sec: 0.0,
                mfu: 0.0,
            };
        }

        // Scale by 1/n_accum
        if n_accum > 1 {
            let scale = 1.0 / n_accum as f32;
            for grad in accumulated_grads.values_mut() {
                B::scale_inplace(grad, scale);
            }
        }

        let grad_norm = clip_grad_norm::<B>(&mut accumulated_grads, self.config.max_grad_norm);

        // Optimizer step
        let mut params: Vec<_> = self
            .model
            .parameters_mut()
            .into_iter()
            .map(|p| {
                let id = p.id;
                let shape = p.shape.clone();
                (id, &mut p.data, shape)
            })
            .collect();
        let mut param_refs: Vec<_> = params
            .iter_mut()
            .map(|(id, data, shape)| (*id, &mut **data, &*shape))
            .collect();
        self.optimizer
            .step(&mut param_refs, &accumulated_grads, &self.no_decay);

        self.step += 1;

        StepMetrics {
            loss: total_loss / n_accum as f32,
            grad_norm,
            lr,
            tokens_per_sec: 0.0,
            mfu: 0.0,
        }
    }

    /// Evaluate on validation batches. Returns mean loss.
    pub fn evaluate(&self, batches: &[Batch]) -> f32 {
        if batches.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0f32;
        for batch in batches {
            let logits = self.model.forward_batch_inference(
                &batch.input_ids,
                batch.batch_size,
                batch.seq_len,
            );
            let loss = ops::cross_entropy(
                &logits,
                &batch.targets,
                batch.batch_size * batch.seq_len,
                self.model_config.vocab_size,
                None,
            );
            total_loss += loss.to_vec()[0];
        }
        total_loss / batches.len() as f32
    }

    /// Run the full training loop.
    ///
    /// # Errors
    ///
    /// Returns an error if data loading fails during training.
    pub fn run(
        &mut self,
        train_loader: &mut DataLoader,
        mut val_loader: Option<&mut DataLoader>,
    ) -> crate::error::Result<()> {
        eprintln!(
            "Starting training: {} total steps, batch_size={}, seq_len={}, grad_accum={}",
            self.config.total_steps,
            self.config.batch_size,
            self.config.seq_len,
            self.config.grad_accum_steps
        );

        let start = std::time::Instant::now();

        while self.step < self.config.total_steps {
            // Collect micro-batches for gradient accumulation
            let mut micro_batches = Vec::with_capacity(self.config.grad_accum_steps);
            for _ in 0..self.config.grad_accum_steps {
                micro_batches.push(train_loader.next_batch()?);
            }

            let metrics = self.train_step(&micro_batches);

            // Logging
            if self.step % self.config.log_interval == 0 || self.step == 1 {
                let elapsed = start.elapsed().as_secs_f64();
                let tokens_per_step = self.config.batch_size
                    * self.config.seq_len
                    * self.config.grad_accum_steps;
                let tokens_per_sec = (self.step * tokens_per_step) as f64 / elapsed;

                // Compute MFU if peak_tflops and n_params are provided
                let mfu = match (self.config.peak_tflops, self.config.n_params) {
                    (Some(peak), Some(n_params)) => {
                        let flops_per_token = 6.0 * n_params as f64;
                        let achieved_flops = tokens_per_sec * flops_per_token;
                        let peak_flops = peak * 1e12;
                        achieved_flops / peak_flops
                    }
                    _ => 0.0,
                };

                if mfu > 0.0 {
                    eprintln!(
                        "step {:>6} | loss {:.4} | ppl {:>8.2} | grad_norm {:.4} | lr {:.2e} | tok/s {:.0} | MFU {:.1}%",
                        self.step,
                        metrics.loss,
                        metrics.loss.exp(),
                        metrics.grad_norm,
                        metrics.lr,
                        tokens_per_sec,
                        mfu * 100.0,
                    );
                } else {
                    eprintln!(
                        "step {:>6} | loss {:.4} | ppl {:>8.2} | grad_norm {:.4} | lr {:.2e} | tok/s {:.0}",
                        self.step,
                        metrics.loss,
                        metrics.loss.exp(),
                        metrics.grad_norm,
                        metrics.lr,
                        tokens_per_sec,
                    );
                }
            }

            // Evaluation
            if let Some(ref mut val) = val_loader.as_deref_mut() {
                if self.config.eval_interval > 0 && self.step % self.config.eval_interval == 0 {
                    let mut val_batches = Vec::new();
                    for _ in 0..10 {
                        if let Ok(batch) = val.next_batch() {
                            val_batches.push(batch);
                        }
                    }
                    let val_loss = self.evaluate(&val_batches);
                    eprintln!("  val_loss {:.4} | val_ppl {:.2}", val_loss, val_loss.exp());
                }
            }

            // Checkpointing
            #[cfg(feature = "safetensors")]
            if self.config.checkpoint_interval > 0
                && self.step % self.config.checkpoint_interval == 0
            {
                let path = self
                    .config
                    .checkpoint_dir
                    .join(format!("step_{}.safetensors", self.step));
                if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                    eprintln!("  warning: cannot create checkpoint dir: {e}");
                } else {
                    match crate::checkpoint::save_checkpoint(
                        &path,
                        &self.model,
                        &self.optimizer,
                        self.step,
                        self.rng.u64(..),
                    ) {
                        Ok(()) => eprintln!("  saved checkpoint: {}", path.display()),
                        Err(e) => eprintln!("  warning: checkpoint save failed: {e}"),
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("Training complete: {} steps in {:.1}s", self.step, elapsed);

        Ok(())
    }
}

/// Add `forward_batch_inference` to `Gpt2Model` for eval (no tape).
impl<B: MathBackend> Gpt2Model<B> {
    pub fn forward_batch_inference(
        &self,
        token_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
    ) -> crate::tensor::Tensor<B> {
        assert_eq!(token_ids.len(), batch_size * seq_len);

        // Position indices reset per sequence
        let positions: Vec<usize> = (0..batch_size).flat_map(|_| 0..seq_len).collect();

        let tok_emb = ops::embedding(
            &self.embedding.token_embedding,
            token_ids,
            self.embedding.vocab_size,
            self.embedding.d_model,
            None,
        );
        let pos_emb = ops::embedding(
            &self.embedding.position_embedding,
            &positions,
            self.embedding.max_seq_len,
            self.embedding.d_model,
            None,
        );
        let mut x = ops::add(&tok_emb, &pos_emb, None);

        for block in &self.blocks {
            x = block.forward_inference(&x);
        }

        x = self.ln_f.forward_inference(&x);

        let total = batch_size * seq_len;
        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            total,
            self.config.d_model,
            self.config.vocab_size,
            false,
            true,
            None,
        )
    }
}

fn merge_grads<B: MathBackend>(dst: &mut Gradients<B>, src: Gradients<B>) {
    for (id, grad) in src {
        if let Some(existing) = dst.get_mut(&id) {
            B::add_inplace(existing, &grad);
        } else {
            dst.insert(id, grad);
        }
    }
}
