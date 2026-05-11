// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF weight-file reader.
//!
//! GGUF is llama.cpp's on-disk format: a single binary file containing a
//! header, key/value metadata table, a tensor descriptor table, and a packed
//! tensor data section. Quantized tensors (Q4_K, Q5_K, Q8_0) are dequantized
//! on demand to `f32` via [`GgufFile::tensor_f32`].
//!
//! Spec reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//! (GGUF v3, current as of llama.cpp ~b3500+).
//!
//! ```no_run
//! # #[cfg(feature = "gguf")] {
//! use scry_llm::gguf::GgufFile;
//! let file = GgufFile::open("model.gguf").unwrap();
//! let arch = file.metadata_string("general.architecture").unwrap();
//! let embed = file.tensor_f32("token_embd.weight").unwrap();
//! # }
//! ```

#![warn(missing_docs)]

mod format;
mod quants;

pub use format::{GgufDtype, GgufMetadataValue, GgufTensorInfo};

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, ScryLlmError};

/// Parsed GGUF file. Owns the raw bytes and exposes typed accessors for
/// metadata and tensors.
///
/// The raw byte buffer is retained so that on-demand tensor reads
/// ([`GgufFile::tensor_f32`]) can dequantize directly out of the original
/// data section without re-reading the file. For an 8B Q4_K_M model that
/// buffer is ~4.8 GiB; callers that load many tensors should do so in a
/// scope that drops the `GgufFile` before peak f32-storage demand hits.
pub struct GgufFile {
    /// Raw bytes of the entire GGUF file (header + metadata + tensor table + data).
    bytes: Vec<u8>,
    /// Byte offset within `bytes` where the tensor data section starts.
    /// All tensor `offset` values in [`GgufTensorInfo`] are relative to this.
    tensor_data_start: usize,
    /// Parsed metadata key/value table.
    metadata: HashMap<String, GgufMetadataValue>,
    /// Parsed tensor descriptor table, keyed by tensor name.
    tensors: HashMap<String, GgufTensorInfo>,
}

impl GgufFile {
    /// Open and parse a GGUF file. Reads the entire file into memory.
    ///
    /// # Errors
    ///
    /// Returns [`ScryLlmError::GgufError`] for I/O failure, magic/version
    /// mismatch, truncated metadata, or unknown tensor dtypes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| ScryLlmError::GgufError(format!("read {}: {e}", path.display())))?;
        Self::from_bytes(bytes)
    }

    /// Parse a GGUF file from an owned byte buffer.
    ///
    /// # Errors
    ///
    /// See [`GgufFile::open`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let parsed = format::parse(&bytes)?;
        Ok(Self {
            tensor_data_start: parsed.tensor_data_start,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            bytes,
        })
    }

    /// All metadata keys present in the file.
    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    /// Look up a metadata value by key.
    pub fn metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.get(key)
    }

    /// Convenience: read a string-valued metadata entry.
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufMetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Convenience: read a `u32` metadata entry, accepting any of the
    /// integer types (u8/u16/u32/u64/i8/i16/i32/i64) provided the value
    /// fits in `u32`.
    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .get(key)?
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
    }

    /// Convenience: read an `f32` metadata entry, accepting any of the
    /// float types.
    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f64().map(|v| v as f32)
    }

    /// All tensor names in the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Look up a tensor descriptor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Read and dequantize a tensor to `f32`.
    ///
    /// Output is row-major in the natural GGUF dimension order (which
    /// matches what llama.cpp writes: for a 2-D matrix shape `[d0, d1]`,
    /// element `(i, j)` lives at index `i * d1 + j`).
    ///
    /// # Errors
    ///
    /// Returns [`ScryLlmError::GgufError`] if the tensor is missing, its
    /// dtype isn't supported, or the on-disk bytes are truncated.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ScryLlmError::GgufError(format!("tensor '{name}' not found")))?;
        let n_elems: usize = info.shape.iter().product();
        let data_start = self.tensor_data_start
            + usize::try_from(info.offset).map_err(|e| {
                ScryLlmError::GgufError(format!("tensor '{name}' offset overflow: {e}"))
            })?;
        let block_bytes = info.dtype.block_size_bytes().ok_or_else(|| {
            ScryLlmError::GgufError(format!(
                "tensor '{name}': dtype {:?} has unknown on-disk layout",
                info.dtype
            ))
        })?;
        let block_elems = info.dtype.block_n_elems().ok_or_else(|| {
            ScryLlmError::GgufError(format!(
                "tensor '{name}': dtype {:?} has unknown on-disk layout",
                info.dtype
            ))
        })?;
        if n_elems % block_elems != 0 {
            return Err(ScryLlmError::GgufError(format!(
                "tensor '{name}': {n_elems} elements not divisible by block size \
                 {block_elems} for dtype {:?}",
                info.dtype
            )));
        }
        let n_blocks = n_elems / block_elems;
        let needed = n_blocks * block_bytes;
        let bytes = self
            .bytes
            .get(data_start..data_start + needed)
            .ok_or_else(|| {
                ScryLlmError::GgufError(format!(
                    "tensor '{name}': data range [{data_start}, {}) exceeds file size {}",
                    data_start + needed,
                    self.bytes.len()
                ))
            })?;
        quants::dequantize(info.dtype, bytes, n_elems)
            .map_err(|e| ScryLlmError::GgufError(format!("tensor '{name}': {e}")))
    }
}
