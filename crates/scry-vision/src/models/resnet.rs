// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ResNet` backbone for image classification and feature extraction.
//!
//! Supports ResNet-18/34 (`BasicBlock`) and ResNet-50/101/152 (Bottleneck).
//! Input: `[3, H, W]` → Output: `[num_classes]` or feature vector.

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::Result;
#[cfg(feature = "safetensors")]
use crate::error::VisionError;
use crate::image::ImageBuffer;
use crate::nn::batchnorm::BatchNorm2d;
use crate::nn::conv2d::Conv2d;
use crate::nn::pool::{AdaptiveAvgPool2d, MaxPool2d};
use crate::nn::relu;
use crate::pipeline::Classify;
use crate::postprocess::classify::{top_k_softmax, Classification};
use crate::transform::resize::{InterpolationMode, Resize};
use crate::transform::to_tensor::ToTensor;
use crate::transform::ImageTransform;

/// Residual block for ResNet-18/34.
///
/// ```text
/// x → Conv3×3(stride) → BN → ReLU → Conv3×3 → BN → (+) → ReLU
///                                                     ↑
/// x → [downsample] ──────────────────────────────────┘
/// ```
pub struct BasicBlock<B: MathBackend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm2d<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm2d<B>,
    /// Downsample skip connection (1×1 conv + BN) when stride > 1 or channels change.
    pub downsample: Option<(Conv2d<B>, BatchNorm2d<B>)>,
    /// True after [`Self::fuse_batchnorms`] has folded each BN into its
    /// preceding conv. Forward then skips the BN ops entirely.
    pub bn_fused: bool,
}

impl<B: MathBackend> BasicBlock<B> {
    /// Expansion factor (output channels = `in_channels` * expansion).
    pub const EXPANSION: usize = 1;

    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2d::square(in_channels, out_channels, 1, stride, 0),
                BatchNorm2d::new(out_channels, 1e-5),
            ))
        } else {
            None
        };
        Self {
            conv1: Conv2d::square(in_channels, out_channels, 3, stride, 1),
            bn1: BatchNorm2d::new(out_channels, 1e-5),
            conv2: Conv2d::square(out_channels, out_channels, 3, 1, 1),
            bn2: BatchNorm2d::new(out_channels, 1e-5),
            downsample,
            bn_fused: false,
        }
    }

    /// Pre-upload all parameter tensors to the backend's device-resident
    /// form. Walks every conv and batchnorm in the block.
    pub fn to_device(&mut self) {
        self.conv1.to_device();
        self.bn1.to_device();
        self.conv2.to_device();
        self.bn2.to_device();
        if let Some((ds_conv, ds_bn)) = &mut self.downsample {
            ds_conv.to_device();
            ds_bn.to_device();
        }
    }

    /// Fold each BN into the conv that precedes it (conv1+bn1, conv2+bn2,
    /// downsample.0+downsample.1) and flip `bn_fused` so forward skips the
    /// BN ops. Must be called before [`Self::to_device`].
    pub fn fuse_batchnorms(&mut self) {
        self.conv1.fold_batchnorm(&self.bn1);
        self.conv2.fold_batchnorm(&self.bn2);
        if let Some((ds_conv, ds_bn)) = &mut self.downsample {
            ds_conv.fold_batchnorm(ds_bn);
        }
        self.bn_fused = true;
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let identity = match self.downsample {
            Some((ref ds_conv, ref ds_bn)) => {
                let ds_out = ds_conv.forward(input);
                if self.bn_fused {
                    ds_out
                } else {
                    ds_bn.forward(&ds_out)
                }
            }
            // Cheap clone — on ScryGpuBackend this is an Arc bump on the
            // device buffer, not a download.
            None => Tensor::<B>::new(B::clone_storage(&input.data), input.shape.clone()),
        };

        let x = if self.bn_fused {
            let x = relu(&self.conv1.forward(input));
            self.conv2.forward(&x)
        } else {
            let x = relu(&self.bn1.forward(&self.conv1.forward(input)));
            self.bn2.forward(&self.conv2.forward(&x))
        };

        // Residual add
        let out_data = B::add(&x.data, &identity.data, &x.shape, &identity.shape, &x.shape);
        relu(&Tensor::new(out_data, x.shape))
    }
}

/// Bottleneck block for ResNet-50/101/152.
///
/// ```text
/// x → Conv1×1 → BN → ReLU → Conv3×3(stride) → BN → ReLU → Conv1×1 → BN → (+) → ReLU
///                                                                             ↑
/// x → [downsample] ─────────────────────────────────────────────────────────┘
/// ```
pub struct Bottleneck<B: MathBackend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm2d<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm2d<B>,
    pub conv3: Conv2d<B>,
    pub bn3: BatchNorm2d<B>,
    pub downsample: Option<(Conv2d<B>, BatchNorm2d<B>)>,
    /// True after [`Self::fuse_batchnorms`] has folded each BN into its
    /// preceding conv. Forward then skips the BN ops entirely.
    pub bn_fused: bool,
}

impl<B: MathBackend> Bottleneck<B> {
    /// Expansion factor.
    pub const EXPANSION: usize = 4;

    /// `mid_channels` is the bottleneck width; output is `mid_channels * 4`.
    pub fn new(in_channels: usize, mid_channels: usize, stride: usize) -> Self {
        let out_channels = mid_channels * Self::EXPANSION;
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2d::square(in_channels, out_channels, 1, stride, 0),
                BatchNorm2d::new(out_channels, 1e-5),
            ))
        } else {
            None
        };
        Self {
            conv1: Conv2d::square(in_channels, mid_channels, 1, 1, 0),
            bn1: BatchNorm2d::new(mid_channels, 1e-5),
            conv2: Conv2d::square(mid_channels, mid_channels, 3, stride, 1),
            bn2: BatchNorm2d::new(mid_channels, 1e-5),
            conv3: Conv2d::square(mid_channels, out_channels, 1, 1, 0),
            bn3: BatchNorm2d::new(out_channels, 1e-5),
            downsample,
            bn_fused: false,
        }
    }

    /// Pre-upload all parameter tensors to the backend's device-resident
    /// form. Walks every conv and batchnorm in the block.
    pub fn to_device(&mut self) {
        self.conv1.to_device();
        self.bn1.to_device();
        self.conv2.to_device();
        self.bn2.to_device();
        self.conv3.to_device();
        self.bn3.to_device();
        if let Some((ds_conv, ds_bn)) = &mut self.downsample {
            ds_conv.to_device();
            ds_bn.to_device();
        }
    }

    /// Fold each BN into the conv that precedes it (conv1+bn1, conv2+bn2,
    /// conv3+bn3, downsample.0+downsample.1) and flip `bn_fused` so forward
    /// skips the BN ops. Must be called before [`Self::to_device`].
    pub fn fuse_batchnorms(&mut self) {
        self.conv1.fold_batchnorm(&self.bn1);
        self.conv2.fold_batchnorm(&self.bn2);
        self.conv3.fold_batchnorm(&self.bn3);
        if let Some((ds_conv, ds_bn)) = &mut self.downsample {
            ds_conv.fold_batchnorm(ds_bn);
        }
        self.bn_fused = true;
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let identity = match self.downsample {
            Some((ref ds_conv, ref ds_bn)) => {
                let ds_out = ds_conv.forward(input);
                if self.bn_fused {
                    ds_out
                } else {
                    ds_bn.forward(&ds_out)
                }
            }
            // Cheap clone — on ScryGpuBackend this is an Arc bump on the
            // device buffer, not a download.
            None => Tensor::<B>::new(B::clone_storage(&input.data), input.shape.clone()),
        };

        let x = if self.bn_fused {
            let x = relu(&self.conv1.forward(input));
            let x = relu(&self.conv2.forward(&x));
            self.conv3.forward(&x)
        } else {
            let x = relu(&self.bn1.forward(&self.conv1.forward(input)));
            let x = relu(&self.bn2.forward(&self.conv2.forward(&x)));
            self.bn3.forward(&self.conv3.forward(&x))
        };

        let out_data = B::add(&x.data, &identity.data, &x.shape, &identity.shape, &x.shape);
        relu(&Tensor::new(out_data, x.shape))
    }
}

/// A stage (layer) of residual blocks.
pub enum ResNetStage<B: MathBackend> {
    Basic(Vec<BasicBlock<B>>),
    Bottleneck(Vec<Bottleneck<B>>),
}

impl<B: MathBackend> ResNetStage<B> {
    /// Pre-upload every block's parameters to the backend's device-resident
    /// form.
    pub fn to_device(&mut self) {
        match self {
            Self::Basic(blocks) => blocks.iter_mut().for_each(BasicBlock::to_device),
            Self::Bottleneck(blocks) => blocks.iter_mut().for_each(Bottleneck::to_device),
        }
    }

    /// Fold BN into preceding conv on every block in this stage.
    /// Must be called before [`Self::to_device`].
    pub fn fuse_batchnorms(&mut self) {
        match self {
            Self::Basic(blocks) => blocks.iter_mut().for_each(BasicBlock::fuse_batchnorms),
            Self::Bottleneck(blocks) => blocks.iter_mut().for_each(Bottleneck::fuse_batchnorms),
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        match self {
            Self::Basic(blocks) => {
                let mut x = blocks[0].forward(input);
                for block in &blocks[1..] {
                    x = block.forward(&x);
                }
                x
            }
            Self::Bottleneck(blocks) => {
                let mut x = blocks[0].forward(input);
                for block in &blocks[1..] {
                    x = block.forward(&x);
                }
                x
            }
        }
    }
}

/// `ResNet` configuration.
#[derive(Clone, Debug)]
pub struct ResNetConfig {
    /// Number of blocks in each of the 4 stages.
    pub layers: [usize; 4],
    /// Use Bottleneck blocks (ResNet-50+) vs `BasicBlock` (ResNet-18/34).
    pub bottleneck: bool,
    /// Number of output classes (0 = feature extractor only, no FC layer).
    pub num_classes: usize,
}

impl ResNetConfig {
    pub fn resnet18(num_classes: usize) -> Self {
        Self {
            layers: [2, 2, 2, 2],
            bottleneck: false,
            num_classes,
        }
    }
    pub fn resnet34(num_classes: usize) -> Self {
        Self {
            layers: [3, 4, 6, 3],
            bottleneck: false,
            num_classes,
        }
    }
    pub fn resnet50(num_classes: usize) -> Self {
        Self {
            layers: [3, 4, 6, 3],
            bottleneck: true,
            num_classes,
        }
    }
    pub fn resnet101(num_classes: usize) -> Self {
        Self {
            layers: [3, 4, 23, 3],
            bottleneck: true,
            num_classes,
        }
    }
    pub fn resnet152(num_classes: usize) -> Self {
        Self {
            layers: [3, 8, 36, 3],
            bottleneck: true,
            num_classes,
        }
    }
}

/// `ResNet` backbone.
///
/// Architecture: Conv7×7 → BN → `ReLU` → `MaxPool` → 4 stages → `AvgPool` → FC
pub struct ResNet<B: MathBackend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm2d<B>,
    pub maxpool: MaxPool2d,
    pub stages: Vec<ResNetStage<B>>,
    pub avgpool: AdaptiveAvgPool2d,
    /// Final FC layer. `None` if `num_classes == 0` (feature extractor mode).
    pub fc_weight: Option<Tensor<B>>,
    pub fc_bias: Option<Tensor<B>>,
    pub config: ResNetConfig,
    /// True after [`Self::fuse_batchnorms`] has folded the stem BN
    /// (`bn1`) into `conv1`. Each stage block tracks its own fold state
    /// in `BasicBlock::bn_fused` / `Bottleneck::bn_fused`.
    pub stem_bn_fused: bool,
}

impl<B: MathBackend> ResNet<B> {
    /// Create a zero-initialized `ResNet` (for testing; real usage loads from checkpoint).
    pub fn new(config: ResNetConfig) -> Self {
        let (stages, feature_dim) = Self::make_stages(&config);

        let (fc_weight, fc_bias) = if config.num_classes > 0 {
            (
                Some(Tensor::from_vec(
                    vec![0.0; feature_dim * config.num_classes],
                    Shape::new(&[feature_dim, config.num_classes]),
                )),
                Some(Tensor::from_vec(
                    vec![0.0; config.num_classes],
                    Shape::new(&[config.num_classes]),
                )),
            )
        } else {
            (None, None)
        };

        Self {
            conv1: Conv2d::new(3, 64, 7, 7, 2, 3),
            bn1: BatchNorm2d::new(64, 1e-5),
            maxpool: MaxPool2d::new(3, 2, 1),
            stages,
            avgpool: AdaptiveAvgPool2d::global(),
            fc_weight,
            fc_bias,
            config,
            stem_bn_fused: false,
        }
    }

    fn make_stages(config: &ResNetConfig) -> (Vec<ResNetStage<B>>, usize) {
        let channels = [64, 128, 256, 512];
        let strides = [1, 2, 2, 2];
        let expansion = if config.bottleneck {
            Bottleneck::<B>::EXPANSION
        } else {
            BasicBlock::<B>::EXPANSION
        };

        let mut stages = Vec::new();
        let mut in_ch = 64;

        for (i, &num_blocks) in config.layers.iter().enumerate() {
            let mid_ch = channels[i];
            let out_ch = mid_ch * expansion;
            let stride = strides[i];

            if config.bottleneck {
                let mut blocks = Vec::new();
                blocks.push(Bottleneck::new(in_ch, mid_ch, stride));
                for _ in 1..num_blocks {
                    blocks.push(Bottleneck::new(out_ch, mid_ch, 1));
                }
                stages.push(ResNetStage::Bottleneck(blocks));
            } else {
                let mut blocks = Vec::new();
                blocks.push(BasicBlock::new(in_ch, mid_ch, stride));
                for _ in 1..num_blocks {
                    blocks.push(BasicBlock::new(mid_ch, mid_ch, 1));
                }
                stages.push(ResNetStage::Basic(blocks));
            }
            in_ch = out_ch;
        }

        (stages, in_ch)
    }

    /// Fold every `BatchNorm` into its preceding conv (stem + every block in
    /// every stage) and flip the corresponding `bn_fused` flags so forward
    /// skips the BN ops. Inference-only optimization — the math is exact.
    ///
    /// Must be called **before** [`Self::to_device`]: folding reads tensor
    /// values via `B::to_vec`, which on GPU backends would force a download
    /// after upload. Idempotency note: calling twice would double-fold and
    /// produce wrong outputs; callers must invoke once. The FC head has no
    /// BN, so it's untouched.
    pub fn fuse_batchnorms(&mut self) {
        self.conv1.fold_batchnorm(&self.bn1);
        self.stem_bn_fused = true;
        for stage in &mut self.stages {
            stage.fuse_batchnorms();
        }
    }

    /// Pre-upload every parameter tensor to the backend's device-resident
    /// form. Walks the stem (conv1, bn1), all four stages, and the FC layer.
    /// No-op on `CpuBackend`; on `ScryGpuBackend` this turns ~50+ per-call
    /// uploads (one per parameter, every forward pass) into a single
    /// up-front upload, so the bench loop measures kernel time rather than
    /// host→device traffic.
    pub fn to_device(&mut self) {
        self.conv1.to_device();
        self.bn1.to_device();
        for stage in &mut self.stages {
            stage.to_device();
        }
        if let Some(w) = &mut self.fc_weight {
            B::to_device_in_place(&mut w.data);
        }
        if let Some(b) = &mut self.fc_bias {
            B::to_device_in_place(&mut b.data);
        }
    }

    /// Feature dimension after global average pooling (before FC).
    pub fn feature_dim(&self) -> usize {
        if self.config.bottleneck {
            2048
        } else {
            512
        }
    }

    /// Forward pass: `[3, H, W]` → `[num_classes]` or `[feature_dim]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        // Stem: conv → [bn] → relu → maxpool. BN is skipped when folded.
        let stem = self.conv1.forward(input);
        let x = if self.stem_bn_fused {
            relu(&stem)
        } else {
            relu(&self.bn1.forward(&stem))
        };
        let x = self.maxpool.forward(&x);

        // 4 residual stages
        let mut x = x;
        for stage in &self.stages {
            x = stage.forward(&x);
        }

        // Global average pool: [C, H, W] → [C, 1, 1]
        let x = self.avgpool.forward(&x);
        let feature_dim = x.shape.dims()[0];

        // FC layer (if classification mode). Avgpool's [C, 1, 1] output has
        // C*1*1 = feature_dim elements in the same row-major layout as
        // [1, feature_dim], so the matmul reads it directly — no copy.
        if let (Some(weight), Some(bias)) = (&self.fc_weight, &self.fc_bias) {
            let logits = B::matmul(
                &x.data,
                &weight.data,
                1,
                feature_dim,
                self.config.num_classes,
                false,
                false,
            );
            let out_shape = Shape::new(&[self.config.num_classes]);
            let bias_shape = Shape::new(&[self.config.num_classes]);
            let result = B::add(&logits, &bias.data, &out_shape, &bias_shape, &out_shape);
            Tensor::<B>::new(result, out_shape)
        } else {
            // Reshape [feature_dim, 1, 1] → [feature_dim] (same storage).
            Tensor::<B>::new(x.data, Shape::new(&[feature_dim]))
        }
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ResNet<B> {
    /// Load a ResNet from a safetensors file with torchvision naming.
    ///
    /// Supports ResNet-18/34 (BasicBlock) and ResNet-50/101/152 (Bottleneck).
    ///
    /// # Naming convention
    ///
    /// Expects PyTorch/torchvision key names:
    /// `conv1.weight`, `bn1.weight`, `layer1.0.conv1.weight`, `fc.weight`, etc.
    ///
    /// # Feature extractor mode
    ///
    /// If `config.num_classes == 0`, the FC layer is skipped even if the file
    /// contains `fc.weight` / `fc.bias`. This lets you use a classification
    /// checkpoint as a feature extractor.
    ///
    /// # Inference perf
    ///
    /// For inference, call [`Self::fuse_batchnorms`] after this and before
    /// [`Self::to_device`] to fold every BN into its preceding conv. The
    /// math is exact and eliminates one kernel + one full activation pass
    /// per BN.
    pub fn from_safetensors(config: ResNetConfig, path: &std::path::Path) -> Result<Self> {
        use crate::checkpoint::{
            load_batchnorm2d, load_conv2d, load_tensor, load_tensor_transposed,
        };

        let file = std::fs::File::open(path)
            .map_err(|e| VisionError::ModelLoad(format!("cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| VisionError::ModelLoad(format!("mmap failed: {e}")))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| VisionError::ModelLoad(format!("safetensors parse failed: {e}")))?;

        let eps = 1e-5;

        // ── Stem ──
        let conv1 = load_conv2d(&tensors, "conv1", 3, 64, 7, 7, 2, 3)?;
        let bn1 = load_batchnorm2d(&tensors, "bn1", 64, eps)?;

        // ── Stages ──
        let channels = [64usize, 128, 256, 512];
        let strides = [1usize, 2, 2, 2];
        let expansion: usize = if config.bottleneck { 4 } else { 1 };

        let mut stages = Vec::new();
        let mut in_ch: usize = 64;

        for (i, &num_blocks) in config.layers.iter().enumerate() {
            let mid_ch = channels[i];
            let out_ch = mid_ch * expansion;
            let stride = strides[i];
            let layer = format!("layer{}", i + 1);

            if config.bottleneck {
                let mut blocks = Vec::new();
                for j in 0..num_blocks {
                    let (blk_in, blk_stride) = if j == 0 { (in_ch, stride) } else { (out_ch, 1) };
                    let prefix = format!("{layer}.{j}");
                    blocks.push(Self::load_bottleneck(
                        &tensors, &prefix, blk_in, mid_ch, blk_stride, eps,
                    )?);
                }
                stages.push(ResNetStage::Bottleneck(blocks));
            } else {
                let mut blocks = Vec::new();
                for j in 0..num_blocks {
                    let (blk_in, blk_stride) = if j == 0 { (in_ch, stride) } else { (mid_ch, 1) };
                    let prefix = format!("{layer}.{j}");
                    blocks.push(Self::load_basic_block(
                        &tensors, &prefix, blk_in, mid_ch, blk_stride, eps,
                    )?);
                }
                stages.push(ResNetStage::Basic(blocks));
            }
            in_ch = out_ch;
        }

        // ── FC head ──
        let feature_dim = in_ch; // 512 for BasicBlock, 2048 for Bottleneck
        let (fc_weight, fc_bias) = if config.num_classes > 0 {
            // PyTorch stores fc.weight as [num_classes, feature_dim]
            // We store as [feature_dim, num_classes], so transpose.
            let w = load_tensor_transposed(&tensors, "fc.weight", config.num_classes, feature_dim)?;
            let b = load_tensor(&tensors, "fc.bias", &[config.num_classes])?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        Ok(Self {
            conv1,
            bn1,
            maxpool: MaxPool2d::new(3, 2, 1),
            stages,
            avgpool: AdaptiveAvgPool2d::global(),
            fc_weight,
            fc_bias,
            config,
            stem_bn_fused: false,
        })
    }

    fn load_basic_block(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        eps: f32,
    ) -> Result<BasicBlock<B>> {
        use crate::checkpoint::{load_batchnorm2d, load_conv2d};

        let conv1 = load_conv2d(
            tensors,
            &format!("{prefix}.conv1"),
            in_channels,
            out_channels,
            3,
            3,
            stride,
            1,
        )?;
        let bn1 = load_batchnorm2d(tensors, &format!("{prefix}.bn1"), out_channels, eps)?;
        let conv2 = load_conv2d(
            tensors,
            &format!("{prefix}.conv2"),
            out_channels,
            out_channels,
            3,
            3,
            1,
            1,
        )?;
        let bn2 = load_batchnorm2d(tensors, &format!("{prefix}.bn2"), out_channels, eps)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = load_conv2d(
                tensors,
                &format!("{prefix}.downsample.0"),
                in_channels,
                out_channels,
                1,
                1,
                stride,
                0,
            )?;
            let ds_bn = load_batchnorm2d(
                tensors,
                &format!("{prefix}.downsample.1"),
                out_channels,
                eps,
            )?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(BasicBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            bn_fused: false,
        })
    }

    fn load_bottleneck(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        mid_channels: usize,
        stride: usize,
        eps: f32,
    ) -> Result<Bottleneck<B>> {
        use crate::checkpoint::{load_batchnorm2d, load_conv2d};

        let out_channels = mid_channels * Bottleneck::<B>::EXPANSION;

        let conv1 = load_conv2d(
            tensors,
            &format!("{prefix}.conv1"),
            in_channels,
            mid_channels,
            1,
            1,
            1,
            0,
        )?;
        let bn1 = load_batchnorm2d(tensors, &format!("{prefix}.bn1"), mid_channels, eps)?;
        let conv2 = load_conv2d(
            tensors,
            &format!("{prefix}.conv2"),
            mid_channels,
            mid_channels,
            3,
            3,
            stride,
            1,
        )?;
        let bn2 = load_batchnorm2d(tensors, &format!("{prefix}.bn2"), mid_channels, eps)?;
        let conv3 = load_conv2d(
            tensors,
            &format!("{prefix}.conv3"),
            mid_channels,
            out_channels,
            1,
            1,
            1,
            0,
        )?;
        let bn3 = load_batchnorm2d(tensors, &format!("{prefix}.bn3"), out_channels, eps)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = load_conv2d(
                tensors,
                &format!("{prefix}.downsample.0"),
                in_channels,
                out_channels,
                1,
                1,
                stride,
                0,
            )?;
            let ds_bn = load_batchnorm2d(
                tensors,
                &format!("{prefix}.downsample.1"),
                out_channels,
                eps,
            )?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Bottleneck {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
            bn_fused: false,
        })
    }
}

/// `ResNet` image classifier implementing the [`Classify`] pipeline trait.
///
/// Handles standard `ImageNet` preprocessing internally:
///
/// 1. **Resize** — bilinear resize to `input_size × input_size` (default 224)
/// 2. **`ToTensor`** — HWC u8 → CHW f32 with `ImageNet` normalization
/// 3. **Forward** — `ResNet` inference → logits
/// 4. **Top-k softmax** — logits → sorted `(class_id, score)` pairs
pub struct ResNetClassifier<B: MathBackend> {
    pub model: ResNet<B>,
    pub input_size: u32,
}

impl<B: MathBackend> ResNetClassifier<B> {
    /// Create a new classifier wrapping a [`ResNet`].
    ///
    /// The model's `num_classes` must be > 0 (classification mode, not feature
    /// extractor mode).
    pub fn new(model: ResNet<B>) -> Self {
        Self {
            model,
            input_size: 224,
        }
    }

    /// Override the input resolution (default: 224).
    #[must_use]
    pub fn with_input_size(mut self, input_size: u32) -> Self {
        self.input_size = input_size;
        self
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ResNetClassifier<B> {
    /// Load a ResNet classifier from a safetensors file.
    pub fn from_safetensors(config: ResNetConfig, path: &std::path::Path) -> Result<Self> {
        let model = ResNet::from_safetensors(config, path)?;
        Ok(Self::new(model))
    }
}

impl<B: MathBackend> Classify for ResNetClassifier<B> {
    fn classify(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        top_k: usize,
    ) -> Result<Vec<Classification>> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        // Resize to model input size
        let resize = Resize::new(
            self.input_size,
            self.input_size,
            InterpolationMode::Bilinear,
        );
        let resized = resize.apply(&img)?;

        // HWC u8 → CHW f32 with ImageNet normalization
        let tensor = ToTensor::imagenet().apply::<B>(&resized);

        // Forward pass → logits
        let logits = self.model.forward(&tensor).to_vec();

        Ok(top_k_softmax(&logits, top_k))
    }
}

impl<B: MathBackend> Module<B> for BasicBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((ref conv, ref bn)) = self.downsample {
            params.extend(conv.parameters());
            params.extend(bn.parameters());
        }
        params
    }
}

impl<B: MathBackend> Module<B> for Bottleneck<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some((ref conv, ref bn)) = self.downsample {
            params.extend(conv.parameters());
            params.extend(bn.parameters());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn basic_block_same_channels() {
        let block = BasicBlock::<CpuBackend>::new(64, 64, 1);
        let input = Tensor::from_vec(vec![0.0; 64 * 8 * 8], Shape::new(&[64, 8, 8]));
        let output = block.forward(&input);
        assert_eq!(output.shape.dims(), &[64, 8, 8]);
    }

    #[test]
    fn basic_block_downsample() {
        let block = BasicBlock::<CpuBackend>::new(64, 128, 2);
        let input = Tensor::from_vec(vec![0.0; 64 * 8 * 8], Shape::new(&[64, 8, 8]));
        let output = block.forward(&input);
        assert_eq!(output.shape.dims(), &[128, 4, 4]);
    }

    #[test]
    fn bottleneck_shape() {
        let block = Bottleneck::<CpuBackend>::new(64, 64, 1);
        let input = Tensor::from_vec(vec![0.0; 64 * 8 * 8], Shape::new(&[64, 8, 8]));
        let output = block.forward(&input);
        // 64 mid → 256 out (expansion=4)
        assert_eq!(output.shape.dims(), &[256, 8, 8]);
    }

    #[test]
    fn bottleneck_downsample() {
        let block = Bottleneck::<CpuBackend>::new(256, 128, 2);
        let input = Tensor::from_vec(vec![0.0; 256 * 8 * 8], Shape::new(&[256, 8, 8]));
        let output = block.forward(&input);
        assert_eq!(output.shape.dims(), &[512, 4, 4]);
    }

    #[test]
    fn resnet18_output_shape() {
        let config = ResNetConfig::resnet18(1000);
        let model = ResNet::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        assert_eq!(output.shape.dims(), &[1000]);
    }

    #[test]
    fn resnet18_feature_extractor() {
        let config = ResNetConfig {
            layers: [2, 2, 2, 2],
            bottleneck: false,
            num_classes: 0, // feature extractor mode
        };
        let model = ResNet::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        assert_eq!(output.shape.dims(), &[512]);
    }

    #[test]
    fn resnet50_output_shape() {
        let config = ResNetConfig::resnet50(1000);
        let model = ResNet::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        assert_eq!(output.shape.dims(), &[1000]);
    }

    #[test]
    fn classifier_returns_top_k() {
        let config = ResNetConfig::resnet18(10);
        let classifier = ResNetClassifier::<CpuBackend>::new(ResNet::new(config));

        // 4×4 RGB image — small enough to be fast with zero weights
        let image = vec![128u8; 4 * 4 * 3];
        let results = classifier.classify(&image, 4, 4, 3).unwrap();

        assert_eq!(results.len(), 3);
        // Scores must be descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // All class IDs must be in range
        for r in &results {
            assert!(r.class_id < 10);
        }
    }

    #[test]
    fn classifier_scores_sum_to_one() {
        let config = ResNetConfig::resnet18(5);
        let classifier = ResNetClassifier::<CpuBackend>::new(ResNet::new(config));

        let image = vec![64u8; 8 * 8 * 3];
        // Request all classes so we can check the full softmax distribution
        let results = classifier.classify(&image, 8, 8, 5).unwrap();

        let sum: f32 = results.iter().map(|r| r.score).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn classifier_top_k_larger_than_classes() {
        let config = ResNetConfig::resnet18(3);
        let classifier = ResNetClassifier::<CpuBackend>::new(ResNet::new(config));

        let image = vec![100u8; 4 * 4 * 3];
        let results = classifier.classify(&image, 4, 4, 100).unwrap();

        // Should return at most num_classes entries
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn classifier_with_custom_input_size() {
        let config = ResNetConfig::resnet18(10);
        let classifier =
            ResNetClassifier::<CpuBackend>::new(ResNet::new(config)).with_input_size(128);

        assert_eq!(classifier.input_size, 128);

        // Should still work — resize handles the dimension change
        let image = vec![128u8; 16 * 16 * 3];
        let results = classifier.classify(&image, 16, 16, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    // ── BN folding equivalence tests ──

    /// Fill a tensor with deterministic pseudo-random f32 values.
    fn fill_random(tensor: &mut Tensor<CpuBackend>, seed: f32) {
        let n = tensor.numel();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13 + seed).sin()).collect();
        *tensor = Tensor::from_vec(v, tensor.shape.clone());
    }

    /// Populate a `BasicBlock` with non-trivial random conv weights and
    /// non-identity BN params so that BN folding is observable.
    fn randomize_basic_block(block: &mut BasicBlock<CpuBackend>, seed: f32) {
        fill_random(&mut block.conv1.weight, seed);
        fill_random(&mut block.conv1.bias, seed + 1.0);
        fill_random(&mut block.bn1.weight, seed + 2.0);
        fill_random(&mut block.bn1.bias, seed + 3.0);
        fill_random(&mut block.bn1.running_mean, seed + 4.0);
        // var must stay positive: shift sin into [0.5, 1.5]
        let n = block.bn1.running_var.numel();
        block.bn1.running_var = Tensor::from_vec(
            (0..n)
                .map(|i| 1.0 + ((i as f32) * 0.07 + seed).sin() * 0.4)
                .collect(),
            block.bn1.running_var.shape.clone(),
        );

        fill_random(&mut block.conv2.weight, seed + 10.0);
        fill_random(&mut block.conv2.bias, seed + 11.0);
        fill_random(&mut block.bn2.weight, seed + 12.0);
        fill_random(&mut block.bn2.bias, seed + 13.0);
        fill_random(&mut block.bn2.running_mean, seed + 14.0);
        let n2 = block.bn2.running_var.numel();
        block.bn2.running_var = Tensor::from_vec(
            (0..n2)
                .map(|i| 1.0 + ((i as f32) * 0.07 + seed + 10.0).sin() * 0.4)
                .collect(),
            block.bn2.running_var.shape.clone(),
        );

        if let Some((ds_conv, ds_bn)) = &mut block.downsample {
            fill_random(&mut ds_conv.weight, seed + 20.0);
            fill_random(&mut ds_conv.bias, seed + 21.0);
            fill_random(&mut ds_bn.weight, seed + 22.0);
            fill_random(&mut ds_bn.bias, seed + 23.0);
            fill_random(&mut ds_bn.running_mean, seed + 24.0);
            let n3 = ds_bn.running_var.numel();
            ds_bn.running_var = Tensor::from_vec(
                (0..n3)
                    .map(|i| 1.0 + ((i as f32) * 0.07 + seed + 20.0).sin() * 0.4)
                    .collect(),
                ds_bn.running_var.shape.clone(),
            );
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        let max_err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < tol,
            "{label}: max error {max_err:.2e} > tol {tol:.2e}"
        );
    }

    fn clone_basic_block(block: &BasicBlock<CpuBackend>) -> BasicBlock<CpuBackend> {
        let in_ch = block.conv1.in_channels;
        let mid_ch = block.conv1.out_channels;
        let stride = block.conv1.stride;
        let mut copy = BasicBlock::<CpuBackend>::new(in_ch, mid_ch, stride);
        copy.conv1.weight = Tensor::from_vec(
            block.conv1.weight.to_vec(),
            block.conv1.weight.shape.clone(),
        );
        copy.conv1.bias =
            Tensor::from_vec(block.conv1.bias.to_vec(), block.conv1.bias.shape.clone());
        copy.bn1.weight =
            Tensor::from_vec(block.bn1.weight.to_vec(), block.bn1.weight.shape.clone());
        copy.bn1.bias = Tensor::from_vec(block.bn1.bias.to_vec(), block.bn1.bias.shape.clone());
        copy.bn1.running_mean = Tensor::from_vec(
            block.bn1.running_mean.to_vec(),
            block.bn1.running_mean.shape.clone(),
        );
        copy.bn1.running_var = Tensor::from_vec(
            block.bn1.running_var.to_vec(),
            block.bn1.running_var.shape.clone(),
        );
        copy.conv2.weight = Tensor::from_vec(
            block.conv2.weight.to_vec(),
            block.conv2.weight.shape.clone(),
        );
        copy.conv2.bias =
            Tensor::from_vec(block.conv2.bias.to_vec(), block.conv2.bias.shape.clone());
        copy.bn2.weight =
            Tensor::from_vec(block.bn2.weight.to_vec(), block.bn2.weight.shape.clone());
        copy.bn2.bias = Tensor::from_vec(block.bn2.bias.to_vec(), block.bn2.bias.shape.clone());
        copy.bn2.running_mean = Tensor::from_vec(
            block.bn2.running_mean.to_vec(),
            block.bn2.running_mean.shape.clone(),
        );
        copy.bn2.running_var = Tensor::from_vec(
            block.bn2.running_var.to_vec(),
            block.bn2.running_var.shape.clone(),
        );
        if let (Some((src_dc, src_db)), Some((dst_dc, dst_db))) =
            (&block.downsample, &mut copy.downsample)
        {
            dst_dc.weight = Tensor::from_vec(src_dc.weight.to_vec(), src_dc.weight.shape.clone());
            dst_dc.bias = Tensor::from_vec(src_dc.bias.to_vec(), src_dc.bias.shape.clone());
            dst_db.weight = Tensor::from_vec(src_db.weight.to_vec(), src_db.weight.shape.clone());
            dst_db.bias = Tensor::from_vec(src_db.bias.to_vec(), src_db.bias.shape.clone());
            dst_db.running_mean = Tensor::from_vec(
                src_db.running_mean.to_vec(),
                src_db.running_mean.shape.clone(),
            );
            dst_db.running_var = Tensor::from_vec(
                src_db.running_var.to_vec(),
                src_db.running_var.shape.clone(),
            );
        }
        copy
    }

    #[test]
    fn basic_block_fuse_matches_unfused_same_channels() {
        let mut reference = BasicBlock::<CpuBackend>::new(16, 16, 1);
        randomize_basic_block(&mut reference, 0.5);
        let mut fused = clone_basic_block(&reference);

        let input = Tensor::from_vec(
            (0..16 * 8 * 8).map(|i| ((i as f32) * 0.21).cos()).collect(),
            Shape::new(&[16, 8, 8]),
        );

        let ref_out = reference.forward(&input);
        fused.fuse_batchnorms();
        assert!(fused.bn_fused);
        let fused_out = fused.forward(&input);

        assert_eq!(fused_out.shape.dims(), ref_out.shape.dims());
        assert_close(
            &fused_out.to_vec(),
            &ref_out.to_vec(),
            1e-3,
            "BasicBlock 16→16",
        );
    }

    #[test]
    fn basic_block_fuse_matches_unfused_with_downsample() {
        let mut reference = BasicBlock::<CpuBackend>::new(8, 16, 2);
        randomize_basic_block(&mut reference, 1.5);
        let mut fused = clone_basic_block(&reference);

        let input = Tensor::from_vec(
            (0..8 * 14 * 14)
                .map(|i| ((i as f32) * 0.17).sin())
                .collect(),
            Shape::new(&[8, 14, 14]),
        );

        let ref_out = reference.forward(&input);
        fused.fuse_batchnorms();
        let fused_out = fused.forward(&input);

        assert_eq!(fused_out.shape.dims(), ref_out.shape.dims());
        assert_close(
            &fused_out.to_vec(),
            &ref_out.to_vec(),
            2e-3,
            "BasicBlock 8→16 stride 2 with downsample",
        );
    }

    #[test]
    fn resnet_fuse_matches_unfused_zero_init() {
        // Zero-init ResNet: weights are 0 but BN running_var=1, eps=1e-5,
        // so folding produces non-trivial conv weights (still 0) but exercises
        // the walker on every block — useful as a smoke test that the walk
        // hits everything without crashing.
        let reference = ResNet::<CpuBackend>::new(ResNetConfig::resnet18(10));
        let mut fused = ResNet::<CpuBackend>::new(ResNetConfig::resnet18(10));

        let input = Tensor::from_vec(vec![0.5; 3 * 32 * 32], Shape::new(&[3, 32, 32]));
        let ref_out = reference.forward(&input);
        fused.fuse_batchnorms();
        assert!(fused.stem_bn_fused);
        let fused_out = fused.forward(&input);

        assert_eq!(fused_out.shape.dims(), ref_out.shape.dims());
        assert_close(
            &fused_out.to_vec(),
            &ref_out.to_vec(),
            1e-4,
            "ResNet-18 zero-init pre/post fuse",
        );
    }

    // ── safetensors round-trip tests ──

    #[cfg(feature = "safetensors")]
    mod safetensors_tests {
        use super::*;
        use std::borrow::Cow;
        use std::collections::HashMap;

        /// Minimal View impl for serializing f32 tensors into safetensors.
        struct F32View {
            data: Vec<u8>,
            shape: Vec<usize>,
        }

        impl F32View {
            fn new(values: &[f32], shape: &[usize]) -> Self {
                let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Self {
                    data,
                    shape: shape.to_vec(),
                }
            }
        }

        impl safetensors::View for F32View {
            fn dtype(&self) -> safetensors::Dtype {
                safetensors::Dtype::F32
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        /// Collect all parameter (name, data, shape) triples from a ResNet
        /// using torchvision naming, and serialize to safetensors bytes.
        fn serialize_resnet(model: &ResNet<CpuBackend>, config: &ResNetConfig) -> Vec<u8> {
            let mut entries: Vec<(String, F32View)> = Vec::new();
            let mut add = |name: &str, tensor: &Tensor<CpuBackend>| {
                entries.push((
                    name.to_string(),
                    F32View::new(&tensor.to_vec(), tensor.shape.dims()),
                ));
            };

            // Stem
            add("conv1.weight", &model.conv1.weight);
            add("bn1.weight", &model.bn1.weight);
            add("bn1.bias", &model.bn1.bias);
            add("bn1.running_mean", &model.bn1.running_mean);
            add("bn1.running_var", &model.bn1.running_var);

            // Stages
            let channels = [64usize, 128, 256, 512];
            let strides = [1usize, 2, 2, 2];
            let expansion: usize = if config.bottleneck { 4 } else { 1 };
            let mut in_ch = 64usize;

            for (i, &num_blocks) in config.layers.iter().enumerate() {
                let mid_ch = channels[i];
                let out_ch = mid_ch * expansion;
                let stride = strides[i];

                for j in 0..num_blocks {
                    let (blk_in, blk_stride) = if j == 0 {
                        (in_ch, stride)
                    } else if config.bottleneck {
                        (out_ch, 1)
                    } else {
                        (mid_ch, 1)
                    };
                    let prefix = format!("layer{}.{j}", i + 1);

                    match &model.stages[i] {
                        ResNetStage::Basic(blocks) => {
                            let b = &blocks[j];
                            add(&format!("{prefix}.conv1.weight"), &b.conv1.weight);
                            add(&format!("{prefix}.bn1.weight"), &b.bn1.weight);
                            add(&format!("{prefix}.bn1.bias"), &b.bn1.bias);
                            add(&format!("{prefix}.bn1.running_mean"), &b.bn1.running_mean);
                            add(&format!("{prefix}.bn1.running_var"), &b.bn1.running_var);
                            add(&format!("{prefix}.conv2.weight"), &b.conv2.weight);
                            add(&format!("{prefix}.bn2.weight"), &b.bn2.weight);
                            add(&format!("{prefix}.bn2.bias"), &b.bn2.bias);
                            add(&format!("{prefix}.bn2.running_mean"), &b.bn2.running_mean);
                            add(&format!("{prefix}.bn2.running_var"), &b.bn2.running_var);
                            if let Some((ref dc, ref db)) = b.downsample {
                                add(&format!("{prefix}.downsample.0.weight"), &dc.weight);
                                add(&format!("{prefix}.downsample.1.weight"), &db.weight);
                                add(&format!("{prefix}.downsample.1.bias"), &db.bias);
                                add(
                                    &format!("{prefix}.downsample.1.running_mean"),
                                    &db.running_mean,
                                );
                                add(
                                    &format!("{prefix}.downsample.1.running_var"),
                                    &db.running_var,
                                );
                            }
                        }
                        ResNetStage::Bottleneck(blocks) => {
                            let b = &blocks[j];
                            add(&format!("{prefix}.conv1.weight"), &b.conv1.weight);
                            add(&format!("{prefix}.bn1.weight"), &b.bn1.weight);
                            add(&format!("{prefix}.bn1.bias"), &b.bn1.bias);
                            add(&format!("{prefix}.bn1.running_mean"), &b.bn1.running_mean);
                            add(&format!("{prefix}.bn1.running_var"), &b.bn1.running_var);
                            add(&format!("{prefix}.conv2.weight"), &b.conv2.weight);
                            add(&format!("{prefix}.bn2.weight"), &b.bn2.weight);
                            add(&format!("{prefix}.bn2.bias"), &b.bn2.bias);
                            add(&format!("{prefix}.bn2.running_mean"), &b.bn2.running_mean);
                            add(&format!("{prefix}.bn2.running_var"), &b.bn2.running_var);
                            add(&format!("{prefix}.conv3.weight"), &b.conv3.weight);
                            add(&format!("{prefix}.bn3.weight"), &b.bn3.weight);
                            add(&format!("{prefix}.bn3.bias"), &b.bn3.bias);
                            add(&format!("{prefix}.bn3.running_mean"), &b.bn3.running_mean);
                            add(&format!("{prefix}.bn3.running_var"), &b.bn3.running_var);
                            if let Some((ref dc, ref db)) = b.downsample {
                                add(&format!("{prefix}.downsample.0.weight"), &dc.weight);
                                add(&format!("{prefix}.downsample.1.weight"), &db.weight);
                                add(&format!("{prefix}.downsample.1.bias"), &db.bias);
                                add(
                                    &format!("{prefix}.downsample.1.running_mean"),
                                    &db.running_mean,
                                );
                                add(
                                    &format!("{prefix}.downsample.1.running_var"),
                                    &db.running_var,
                                );
                            }
                        }
                    }
                    let _ = (blk_in, blk_stride);
                }
                in_ch = out_ch;
            }

            // FC (PyTorch stores [num_classes, feature_dim] — transposed from ours)
            if let (Some(w), Some(b)) = (&model.fc_weight, &model.fc_bias) {
                let our_data = w.to_vec(); // [feature_dim, num_classes] row-major
                let feature_dim = if config.bottleneck { 2048 } else { 512 };
                let nc = config.num_classes;
                let transposed = crate::checkpoint::transpose_2d(&our_data, feature_dim, nc);
                entries.push((
                    "fc.weight".to_string(),
                    F32View::new(&transposed, &[nc, feature_dim]),
                ));
                entries.push((
                    "fc.bias".to_string(),
                    F32View::new(&b.to_vec(), b.shape.dims()),
                ));
            }

            let info: Option<HashMap<String, String>> = None;
            safetensors::serialize(entries, &info).unwrap()
        }

        /// Compare two tensors element-wise.
        fn tensors_equal(a: &Tensor<CpuBackend>, b: &Tensor<CpuBackend>, name: &str) {
            assert_eq!(a.shape.dims(), b.shape.dims(), "shape mismatch for {name}");
            let av = a.to_vec();
            let bv = b.to_vec();
            for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!((x - y).abs() < 1e-6, "{name}[{i}]: {x} vs {y}",);
            }
        }

        fn assert_resnet_equal(a: &ResNet<CpuBackend>, b: &ResNet<CpuBackend>) {
            // Stem
            tensors_equal(&a.conv1.weight, &b.conv1.weight, "conv1.weight");
            tensors_equal(&a.bn1.weight, &b.bn1.weight, "bn1.weight");
            tensors_equal(&a.bn1.bias, &b.bn1.bias, "bn1.bias");
            tensors_equal(&a.bn1.running_mean, &b.bn1.running_mean, "bn1.running_mean");
            tensors_equal(&a.bn1.running_var, &b.bn1.running_var, "bn1.running_var");

            // Stages — compare block by block
            assert_eq!(a.stages.len(), b.stages.len());
            for (si, (sa, sb)) in a.stages.iter().zip(b.stages.iter()).enumerate() {
                match (sa, sb) {
                    (ResNetStage::Basic(ba), ResNetStage::Basic(bb)) => {
                        assert_eq!(ba.len(), bb.len());
                        for (bi, (a_blk, b_blk)) in ba.iter().zip(bb.iter()).enumerate() {
                            let p = format!("layer{}.{bi}", si + 1);
                            tensors_equal(
                                &a_blk.conv1.weight,
                                &b_blk.conv1.weight,
                                &format!("{p}.conv1.w"),
                            );
                            tensors_equal(
                                &a_blk.bn1.weight,
                                &b_blk.bn1.weight,
                                &format!("{p}.bn1.w"),
                            );
                            tensors_equal(
                                &a_blk.conv2.weight,
                                &b_blk.conv2.weight,
                                &format!("{p}.conv2.w"),
                            );
                            tensors_equal(
                                &a_blk.bn2.weight,
                                &b_blk.bn2.weight,
                                &format!("{p}.bn2.w"),
                            );
                            match (&a_blk.downsample, &b_blk.downsample) {
                                (Some((ac, ab)), Some((bc, bb))) => {
                                    tensors_equal(
                                        &ac.weight,
                                        &bc.weight,
                                        &format!("{p}.ds.conv.w"),
                                    );
                                    tensors_equal(&ab.weight, &bb.weight, &format!("{p}.ds.bn.w"));
                                }
                                (None, None) => {}
                                _ => panic!("downsample mismatch at {p}"),
                            }
                        }
                    }
                    (ResNetStage::Bottleneck(ba), ResNetStage::Bottleneck(bb)) => {
                        assert_eq!(ba.len(), bb.len());
                        for (bi, (a_blk, b_blk)) in ba.iter().zip(bb.iter()).enumerate() {
                            let p = format!("layer{}.{bi}", si + 1);
                            tensors_equal(
                                &a_blk.conv1.weight,
                                &b_blk.conv1.weight,
                                &format!("{p}.conv1.w"),
                            );
                            tensors_equal(
                                &a_blk.conv2.weight,
                                &b_blk.conv2.weight,
                                &format!("{p}.conv2.w"),
                            );
                            tensors_equal(
                                &a_blk.conv3.weight,
                                &b_blk.conv3.weight,
                                &format!("{p}.conv3.w"),
                            );
                            tensors_equal(
                                &a_blk.bn1.weight,
                                &b_blk.bn1.weight,
                                &format!("{p}.bn1.w"),
                            );
                            tensors_equal(
                                &a_blk.bn2.weight,
                                &b_blk.bn2.weight,
                                &format!("{p}.bn2.w"),
                            );
                            tensors_equal(
                                &a_blk.bn3.weight,
                                &b_blk.bn3.weight,
                                &format!("{p}.bn3.w"),
                            );
                            match (&a_blk.downsample, &b_blk.downsample) {
                                (Some((ac, ab)), Some((bc, bb))) => {
                                    tensors_equal(
                                        &ac.weight,
                                        &bc.weight,
                                        &format!("{p}.ds.conv.w"),
                                    );
                                    tensors_equal(&ab.weight, &bb.weight, &format!("{p}.ds.bn.w"));
                                }
                                (None, None) => {}
                                _ => panic!("downsample mismatch at {p}"),
                            }
                        }
                    }
                    _ => panic!("stage type mismatch at stage {si}"),
                }
            }

            // FC
            match (&a.fc_weight, &b.fc_weight) {
                (Some(aw), Some(bw)) => tensors_equal(aw, bw, "fc.weight"),
                (None, None) => {}
                _ => panic!("fc_weight presence mismatch"),
            }
            match (&a.fc_bias, &b.fc_bias) {
                (Some(ab), Some(bb)) => tensors_equal(ab, bb, "fc.bias"),
                (None, None) => {}
                _ => panic!("fc_bias presence mismatch"),
            }
        }

        /// Write bytes to a unique temp file and return the path.
        fn write_temp(data: &[u8], label: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir();
            let path = dir.join(format!(
                "scry_test_{}_{label}.safetensors",
                std::process::id(),
            ));
            std::fs::write(&path, data).unwrap();
            path
        }

        #[test]
        fn roundtrip_resnet18() {
            let config = ResNetConfig::resnet18(10);
            let original = ResNet::<CpuBackend>::new(config.clone());

            let bytes = serialize_resnet(&original, &config);
            let path = write_temp(&bytes, "r18");
            let loaded = ResNet::<CpuBackend>::from_safetensors(config, &path).unwrap();
            std::fs::remove_file(&path).ok();

            assert_resnet_equal(&original, &loaded);
        }

        #[test]
        fn roundtrip_resnet50() {
            let config = ResNetConfig::resnet50(100);
            let original = ResNet::<CpuBackend>::new(config.clone());

            let bytes = serialize_resnet(&original, &config);
            let path = write_temp(&bytes, "r50");
            let loaded = ResNet::<CpuBackend>::from_safetensors(config, &path).unwrap();
            std::fs::remove_file(&path).ok();

            assert_resnet_equal(&original, &loaded);
        }

        #[test]
        fn roundtrip_feature_extractor() {
            // num_classes=0: FC layer should be skipped
            let config = ResNetConfig {
                layers: [2, 2, 2, 2],
                bottleneck: false,
                num_classes: 0,
            };
            let original = ResNet::<CpuBackend>::new(config.clone());

            let bytes = serialize_resnet(&original, &config);
            let path = write_temp(&bytes, "feat");
            let loaded = ResNet::<CpuBackend>::from_safetensors(config, &path).unwrap();
            std::fs::remove_file(&path).ok();

            assert!(loaded.fc_weight.is_none());
            assert!(loaded.fc_bias.is_none());
            assert_resnet_equal(&original, &loaded);
        }

        #[test]
        fn missing_weight_produces_error() {
            // Empty safetensors file — should fail on first weight
            let info: Option<HashMap<String, String>> = None;
            let bytes = safetensors::serialize(Vec::<(String, F32View)>::new(), &info).unwrap();
            let path = write_temp(&bytes, "empty");

            let config = ResNetConfig::resnet18(10);
            let result = ResNet::<CpuBackend>::from_safetensors(config, &path);
            std::fs::remove_file(&path).ok();

            match result {
                Err(e) => {
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("conv1.weight"),
                        "error should mention conv1.weight: {msg}"
                    );
                }
                Ok(_) => panic!("should fail on empty safetensors"),
            }
        }

        #[test]
        fn bad_path_produces_error() {
            let config = ResNetConfig::resnet18(10);
            let result = ResNet::<CpuBackend>::from_safetensors(
                config,
                std::path::Path::new("/nonexistent/model.safetensors"),
            );
            assert!(result.is_err());
        }
    }
}
