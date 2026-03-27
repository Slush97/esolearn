// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors checkpoint loading helpers.
//!
//! Provides dtype-agnostic tensor loading from safetensors files.
//! Requires `feature = "safetensors"`.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::{Result, VisionError};

/// Load a named tensor from a safetensors file, with dtype auto-detection.
///
/// Automatically converts F16, BF16, and F32 tensors to `f32`.
/// Validates that the loaded data matches the expected `shape`.
pub fn load_tensor<B: MathBackend>(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Tensor<B>> {
    let data = load_f32(tensors, name)?;
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(VisionError::ShapeMismatch {
            name: name.to_string(),
            expected: shape.to_vec(),
            got: vec![data.len()],
        });
    }
    Ok(Tensor::from_vec(data, Shape::new(shape)))
}

/// Load a 2D weight tensor and transpose it from `[rows, cols]` to `[cols, rows]`.
///
/// Useful for linear layer weights stored in HuggingFace format `[out, in]`
/// that need to be transposed to `[in, out]` for scry-llm's Linear.
pub fn load_tensor_transposed<B: MathBackend>(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Tensor<B>> {
    let data = load_f32(tensors, name)?;
    if data.len() != rows * cols {
        return Err(VisionError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![rows, cols],
            got: vec![data.len()],
        });
    }
    let transposed = transpose_2d(&data, rows, cols);
    Ok(Tensor::from_vec(transposed, Shape::new(&[cols, rows])))
}

/// Load raw f32 data from a named tensor, with dtype auto-detection.
pub fn load_f32(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<Vec<f32>> {
    let t = tensors
        .tensor(name)
        .map_err(|_| VisionError::MissingWeight(name.to_string()))?;
    match t.dtype() {
        safetensors::Dtype::F16 => Ok(f16_bytes_to_f32(t.data())),
        safetensors::Dtype::BF16 => Ok(bf16_bytes_to_f32(t.data())),
        safetensors::Dtype::F32 => Ok(f32_bytes_to_f32(t.data())),
        other => Err(VisionError::ModelLoad(format!(
            "unsupported dtype {other:?} for tensor '{name}'"
        ))),
    }
}

/// Transpose a 2D matrix stored row-major: `[rows, cols]` → `[cols, rows]`.
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Convert little-endian f16 bytes to f32.
fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Convert little-endian bf16 bytes to f32.
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Convert little-endian f32 bytes to f32.
fn f32_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d_basic() {
        // [[1, 2, 3], [4, 5, 6]] → [[1, 4], [2, 5], [3, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_2d(&data, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_square() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = transpose_2d(&data, 2, 2);
        assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn f32_roundtrip() {
        let values = vec![1.0f32, -2.5, 0.0, 3.14];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = f32_bytes_to_f32(&bytes);
        assert_eq!(result, values);
    }

    #[test]
    fn f16_conversion() {
        let values = [1.0f32, -1.0, 0.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let result = f16_bytes_to_f32(&bytes);
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn bf16_conversion() {
        let values = [1.0f32, -1.0, 0.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let result = bf16_bytes_to_f32(&bytes);
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }
}
