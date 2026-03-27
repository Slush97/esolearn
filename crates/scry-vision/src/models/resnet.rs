// SPDX-License-Identifier: MIT OR Apache-2.0
//! ResNet backbone for image classification and feature extraction.
//!
//! Supports ResNet-18/34 (BasicBlock) and ResNet-50/101/152 (Bottleneck).
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
}

impl<B: MathBackend> BasicBlock<B> {
    /// Expansion factor (output channels = in_channels * expansion).
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
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let identity = match self.downsample {
            Some((ref ds_conv, ref ds_bn)) => ds_bn.forward(&ds_conv.forward(input)),
            None => Tensor::from_vec(input.to_vec(), input.shape.clone()),
        };

        let x = relu(&self.bn1.forward(&self.conv1.forward(input)));
        let x = self.bn2.forward(&self.conv2.forward(&x));

        // Residual add
        let out_data = B::add(
            &x.data,
            &identity.data,
            &x.shape,
            &identity.shape,
            &x.shape,
        );
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
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let identity = match self.downsample {
            Some((ref ds_conv, ref ds_bn)) => ds_bn.forward(&ds_conv.forward(input)),
            None => Tensor::from_vec(input.to_vec(), input.shape.clone()),
        };

        let x = relu(&self.bn1.forward(&self.conv1.forward(input)));
        let x = relu(&self.bn2.forward(&self.conv2.forward(&x)));
        let x = self.bn3.forward(&self.conv3.forward(&x));

        let out_data = B::add(
            &x.data,
            &identity.data,
            &x.shape,
            &identity.shape,
            &x.shape,
        );
        relu(&Tensor::new(out_data, x.shape))
    }
}

/// A stage (layer) of residual blocks.
pub enum ResNetStage<B: MathBackend> {
    Basic(Vec<BasicBlock<B>>),
    Bottleneck(Vec<Bottleneck<B>>),
}

impl<B: MathBackend> ResNetStage<B> {
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

/// ResNet configuration.
#[derive(Clone, Debug)]
pub struct ResNetConfig {
    /// Number of blocks in each of the 4 stages.
    pub layers: [usize; 4],
    /// Use Bottleneck blocks (ResNet-50+) vs BasicBlock (ResNet-18/34).
    pub bottleneck: bool,
    /// Number of output classes (0 = feature extractor only, no FC layer).
    pub num_classes: usize,
}

impl ResNetConfig {
    pub fn resnet18(num_classes: usize) -> Self {
        Self { layers: [2, 2, 2, 2], bottleneck: false, num_classes }
    }
    pub fn resnet34(num_classes: usize) -> Self {
        Self { layers: [3, 4, 6, 3], bottleneck: false, num_classes }
    }
    pub fn resnet50(num_classes: usize) -> Self {
        Self { layers: [3, 4, 6, 3], bottleneck: true, num_classes }
    }
    pub fn resnet101(num_classes: usize) -> Self {
        Self { layers: [3, 4, 23, 3], bottleneck: true, num_classes }
    }
    pub fn resnet152(num_classes: usize) -> Self {
        Self { layers: [3, 8, 36, 3], bottleneck: true, num_classes }
    }
}

/// ResNet backbone.
///
/// Architecture: Conv7×7 → BN → ReLU → MaxPool → 4 stages → AvgPool → FC
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
}

impl<B: MathBackend> ResNet<B> {
    /// Create a zero-initialized ResNet (for testing; real usage loads from checkpoint).
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

    /// Feature dimension after global average pooling (before FC).
    pub fn feature_dim(&self) -> usize {
        if self.config.bottleneck { 2048 } else { 512 }
    }

    /// Forward pass: `[3, H, W]` → `[num_classes]` or `[feature_dim]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        // Stem: conv → bn → relu → maxpool
        let x = relu(&self.bn1.forward(&self.conv1.forward(input)));
        let x = self.maxpool.forward(&x);

        // 4 residual stages
        let mut x = x;
        for stage in &self.stages {
            x = stage.forward(&x);
        }

        // Global average pool: [C, H, W] → [C, 1, 1]
        let x = self.avgpool.forward(&x);
        let feature_dim = x.shape.dims()[0];
        let features = Tensor::<B>::from_vec(x.to_vec(), Shape::new(&[feature_dim]));

        // FC layer (if classification mode)
        if let (Some(weight), Some(bias)) = (&self.fc_weight, &self.fc_bias) {
            // features[1, feature_dim] @ weight[feature_dim, num_classes] + bias
            let features_2d = B::from_vec(
                features.to_vec(),
                &Shape::new(&[1, feature_dim]),
            );
            let logits = B::matmul(
                &features_2d,
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
            Tensor::new(result, out_shape)
        } else {
            features
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
    pub fn from_safetensors(
        config: ResNetConfig,
        path: &std::path::Path,
    ) -> Result<Self> {
        use crate::checkpoint::{load_batchnorm2d, load_conv2d, load_tensor, load_tensor_transposed};

        let file = std::fs::File::open(path).map_err(|e| {
            VisionError::ModelLoad(format!("cannot open {}: {e}", path.display()))
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            VisionError::ModelLoad(format!("mmap failed: {e}"))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| {
            VisionError::ModelLoad(format!("safetensors parse failed: {e}"))
        })?;

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
                    let (blk_in, blk_stride) = if j == 0 {
                        (in_ch, stride)
                    } else {
                        (out_ch, 1)
                    };
                    let prefix = format!("{layer}.{j}");
                    blocks.push(Self::load_bottleneck(
                        &tensors, &prefix, blk_in, mid_ch, blk_stride, eps,
                    )?);
                }
                stages.push(ResNetStage::Bottleneck(blocks));
            } else {
                let mut blocks = Vec::new();
                for j in 0..num_blocks {
                    let (blk_in, blk_stride) = if j == 0 {
                        (in_ch, stride)
                    } else {
                        (mid_ch, 1)
                    };
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
            let w = load_tensor_transposed(
                &tensors, "fc.weight", config.num_classes, feature_dim,
            )?;
            let b = load_tensor(
                &tensors, "fc.bias", &[config.num_classes],
            )?;
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
            tensors, &format!("{prefix}.conv1"),
            in_channels, out_channels, 3, 3, stride, 1,
        )?;
        let bn1 = load_batchnorm2d(tensors, &format!("{prefix}.bn1"), out_channels, eps)?;
        let conv2 = load_conv2d(
            tensors, &format!("{prefix}.conv2"),
            out_channels, out_channels, 3, 3, 1, 1,
        )?;
        let bn2 = load_batchnorm2d(tensors, &format!("{prefix}.bn2"), out_channels, eps)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = load_conv2d(
                tensors, &format!("{prefix}.downsample.0"),
                in_channels, out_channels, 1, 1, stride, 0,
            )?;
            let ds_bn = load_batchnorm2d(
                tensors, &format!("{prefix}.downsample.1"), out_channels, eps,
            )?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(BasicBlock { conv1, bn1, conv2, bn2, downsample })
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
            tensors, &format!("{prefix}.conv1"),
            in_channels, mid_channels, 1, 1, 1, 0,
        )?;
        let bn1 = load_batchnorm2d(tensors, &format!("{prefix}.bn1"), mid_channels, eps)?;
        let conv2 = load_conv2d(
            tensors, &format!("{prefix}.conv2"),
            mid_channels, mid_channels, 3, 3, stride, 1,
        )?;
        let bn2 = load_batchnorm2d(tensors, &format!("{prefix}.bn2"), mid_channels, eps)?;
        let conv3 = load_conv2d(
            tensors, &format!("{prefix}.conv3"),
            mid_channels, out_channels, 1, 1, 1, 0,
        )?;
        let bn3 = load_batchnorm2d(tensors, &format!("{prefix}.bn3"), out_channels, eps)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = load_conv2d(
                tensors, &format!("{prefix}.downsample.0"),
                in_channels, out_channels, 1, 1, stride, 0,
            )?;
            let ds_bn = load_batchnorm2d(
                tensors, &format!("{prefix}.downsample.1"), out_channels, eps,
            )?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Bottleneck { conv1, bn1, conv2, bn2, conv3, bn3, downsample })
    }
}

/// ResNet image classifier implementing the [`Classify`] pipeline trait.
///
/// Handles standard ImageNet preprocessing internally:
///
/// 1. **Resize** — bilinear resize to `input_size × input_size` (default 224)
/// 2. **ToTensor** — HWC u8 → CHW f32 with ImageNet normalization
/// 3. **Forward** — ResNet inference → logits
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
    pub fn from_safetensors(
        config: ResNetConfig,
        path: &std::path::Path,
    ) -> Result<Self> {
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
        let resize = Resize::new(self.input_size, self.input_size, InterpolationMode::Bilinear);
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
        let classifier = ResNetClassifier::<CpuBackend>::new(ResNet::new(config))
            .with_input_size(128);

        assert_eq!(classifier.input_size, 128);

        // Should still work — resize handles the dimension change
        let image = vec![128u8; 16 * 16 * 3];
        let results = classifier.classify(&image, 16, 16, 5).unwrap();
        assert_eq!(results.len(), 5);
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
                Self { data, shape: shape.to_vec() }
            }
        }

        impl safetensors::View for F32View {
            fn dtype(&self) -> safetensors::Dtype { safetensors::Dtype::F32 }
            fn shape(&self) -> &[usize] { &self.shape }
            fn data(&self) -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize { self.data.len() }
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
                                add(&format!("{prefix}.downsample.1.running_mean"), &db.running_mean);
                                add(&format!("{prefix}.downsample.1.running_var"), &db.running_var);
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
                                add(&format!("{prefix}.downsample.1.running_mean"), &db.running_mean);
                                add(&format!("{prefix}.downsample.1.running_var"), &db.running_var);
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
                assert!(
                    (x - y).abs() < 1e-6,
                    "{name}[{i}]: {x} vs {y}",
                );
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
                            tensors_equal(&a_blk.conv1.weight, &b_blk.conv1.weight, &format!("{p}.conv1.w"));
                            tensors_equal(&a_blk.bn1.weight, &b_blk.bn1.weight, &format!("{p}.bn1.w"));
                            tensors_equal(&a_blk.conv2.weight, &b_blk.conv2.weight, &format!("{p}.conv2.w"));
                            tensors_equal(&a_blk.bn2.weight, &b_blk.bn2.weight, &format!("{p}.bn2.w"));
                            match (&a_blk.downsample, &b_blk.downsample) {
                                (Some((ac, ab)), Some((bc, bb))) => {
                                    tensors_equal(&ac.weight, &bc.weight, &format!("{p}.ds.conv.w"));
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
                            tensors_equal(&a_blk.conv1.weight, &b_blk.conv1.weight, &format!("{p}.conv1.w"));
                            tensors_equal(&a_blk.conv2.weight, &b_blk.conv2.weight, &format!("{p}.conv2.w"));
                            tensors_equal(&a_blk.conv3.weight, &b_blk.conv3.weight, &format!("{p}.conv3.w"));
                            tensors_equal(&a_blk.bn1.weight, &b_blk.bn1.weight, &format!("{p}.bn1.w"));
                            tensors_equal(&a_blk.bn2.weight, &b_blk.bn2.weight, &format!("{p}.bn2.w"));
                            tensors_equal(&a_blk.bn3.weight, &b_blk.bn3.weight, &format!("{p}.bn3.w"));
                            match (&a_blk.downsample, &b_blk.downsample) {
                                (Some((ac, ab)), Some((bc, bb))) => {
                                    tensors_equal(&ac.weight, &bc.weight, &format!("{p}.ds.conv.w"));
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
            let config = ResNetConfig { layers: [2, 2, 2, 2], bottleneck: false, num_classes: 0 };
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
            let bytes = safetensors::serialize(
                Vec::<(String, F32View)>::new(),
                &info,
            ).unwrap();
            let path = write_temp(&bytes, "empty");

            let config = ResNetConfig::resnet18(10);
            let result = ResNet::<CpuBackend>::from_safetensors(config, &path);
            std::fs::remove_file(&path).ok();

            match result {
                Err(e) => {
                    let msg = format!("{e}");
                    assert!(msg.contains("conv1.weight"), "error should mention conv1.weight: {msg}");
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
