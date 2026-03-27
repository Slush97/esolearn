// SPDX-License-Identifier: MIT OR Apache-2.0
//! ResNet backbone for image classification and feature extraction.
//!
//! Supports ResNet-18/34 (BasicBlock) and ResNet-50/101/152 (Bottleneck).
//! Input: `[3, H, W]` → Output: `[num_classes]` or feature vector.

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::nn::batchnorm::BatchNorm2d;
use crate::nn::conv2d::Conv2d;
use crate::nn::pool::{AdaptiveAvgPool2d, MaxPool2d};
use crate::nn::relu;

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
}
