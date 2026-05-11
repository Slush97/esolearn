// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF v3 binary format parser.
//!
//! Layout (all numbers little-endian):
//! ```text
//! [magic: 4 bytes "GGUF"]
//! [version: u32]                        // 3 supported
//! [tensor_count: u64]
//! [metadata_kv_count: u64]
//! [metadata_kv * metadata_kv_count]
//! [tensor_info * tensor_count]
//! [pad to alignment (default 32)]
//! [tensor_data]
//! ```
//! A `metadata_kv` is `(string key, u32 value_type, value)`.
//! A `tensor_info` is `(string name, u32 n_dims, u64 dims[n_dims], u32 type, u64 offset)`.
//! Strings are `(u64 length, utf8 bytes)`; no null terminator.

use std::collections::HashMap;

use crate::error::{Result, ScryLlmError};

/// Tensor dtype on disk.
///
/// The variants the M14 GGUF loader can actually dequantize are enumerated
/// explicitly. Other GGML type IDs (Q2_K, Q3_K, Q6_K, IQ variants, …)
/// parse as [`GgufDtype::Other`] so the tensor table can be read in full;
/// attempting to materialize such a tensor via
/// [`GgufFile::tensor_f32`](super::GgufFile::tensor_f32) fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    /// IEEE-754 binary32, native byte order.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// 32-element blocks of i8 with a single f16 scale.
    Q8_0,
    /// K-quant: 256-element superblock, 4-bit weights, 8 sub-blocks each
    /// with a 6-bit scale + 6-bit min derived from two f16 super-scalars.
    Q4K,
    /// K-quant: 256-element superblock, 5-bit weights (4-bit qs + 1-bit qh),
    /// otherwise the same scale/min layout as Q4_K.
    Q5K,
    /// K-quant: 256-element superblock, 6-bit weights. Common for `output`
    /// and `token_embd` in `Q4_K_M` / `Q5_K_M` mixed quants.
    Q6K,
    /// IEEE-754 brain-float16.
    BF16,
    /// A recognized but unsupported-for-dequant GGML type id. The block
    /// size is looked up from a static table; if the id is also unknown
    /// to the table, [`block_size_bytes`](Self::block_size_bytes) returns
    /// `None` and any read of such a tensor errors.
    Other(u32),
}

impl GgufDtype {
    pub(super) fn from_ggml_type(t: u32) -> Self {
        match t {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// Bytes per on-disk block. `None` for [`GgufDtype::Other`] ids whose
    /// layout we don't know (so we can't compute tensor data sizes for them).
    pub(super) fn block_size_bytes(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::Q8_0 => Some(34),
            Self::Q4K => Some(144),
            Self::Q5K => Some(176),
            // f16 d (2) + 128 qs (128) + 64 qh (64) + 16 scales (16) = 210.
            Self::Q6K => Some(210),
            Self::Other(id) => ggml_block_size(id),
        }
    }

    /// Elements per on-disk block. `None` for unknown [`GgufDtype::Other`] ids.
    pub(super) fn block_n_elems(self) -> Option<usize> {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => Some(1),
            Self::Q8_0 => Some(32),
            Self::Q4K | Self::Q5K | Self::Q6K => Some(256),
            Self::Other(id) => ggml_block_n_elems(id),
        }
    }
}

/// Static table of `(block_bytes, block_elems)` for known GGML types we
/// don't dequant ourselves. Lets the parser still compute byte ranges so
/// callers can probe the tensor table.
fn ggml_block_info(id: u32) -> Option<(usize, usize)> {
    // Source: llama.cpp/ggml.h type_traits[] table.
    match id {
        2 => Some((18, 32)),    // Q4_0
        3 => Some((20, 32)),    // Q4_1
        6 => Some((22, 32)),    // Q5_0
        7 => Some((24, 32)),    // Q5_1
        9 => Some((36, 32)),    // Q8_1
        10 => Some((84, 256)),  // Q2_K
        11 => Some((110, 256)), // Q3_K
        15 => Some((292, 256)), // Q8_K (used internally, rare on disk)
        _ => None,
    }
}

fn ggml_block_size(id: u32) -> Option<usize> {
    ggml_block_info(id).map(|(bytes, _)| bytes)
}

fn ggml_block_n_elems(id: u32) -> Option<usize> {
    ggml_block_info(id).map(|(_, elems)| elems)
}

/// One tensor's descriptor as recorded in the GGUF tensor table.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Dimensions in GGUF order. For a 2-D matrix this is `[d0, d1]` with
    /// row-major layout — element `(i, j)` lives at flat index `i * d1 + j`.
    pub shape: Vec<usize>,
    /// On-disk dtype.
    pub dtype: GgufDtype,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

/// A metadata value. GGUF supports primitives, strings, and (typed) arrays.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// `u8` value.
    U8(u8),
    /// `i8` value.
    I8(i8),
    /// `u16` value.
    U16(u16),
    /// `i16` value.
    I16(i16),
    /// `u32` value.
    U32(u32),
    /// `i32` value.
    I32(i32),
    /// `f32` value.
    F32(f32),
    /// `bool` value.
    Bool(bool),
    /// UTF-8 string value.
    String(String),
    /// `u64` value.
    U64(u64),
    /// `i64` value.
    I64(i64),
    /// `f64` value.
    F64(f64),
    /// Typed array. The first field is the element type id (the same GGUF
    /// type tag as for scalar metadata), the second is the elements.
    Array(Box<[GgufMetadataValue]>),
}

impl GgufMetadataValue {
    /// Integer accessor — returns the value as `u64` if it is any integer
    /// type and non-negative.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::U16(v) => Some(u64::from(v)),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::U32(v) => Some(u64::from(v)),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I64(v) if v >= 0 => Some(v as u64),
            Self::Bool(b) => Some(u64::from(b)),
            _ => None,
        }
    }

    /// Float accessor — returns the value as `f64` if it is any float type
    /// or an integer that round-trips losslessly.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
}

pub(super) struct ParsedFile {
    pub tensor_data_start: usize,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
}

pub(super) fn parse(bytes: &[u8]) -> Result<ParsedFile> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_bytes(4)?;
    if magic != b"GGUF" {
        return Err(ScryLlmError::GgufError(format!(
            "bad magic: expected b\"GGUF\", got {magic:?}"
        )));
    }
    let version = cur.read_u32()?;
    if version != 3 {
        return Err(ScryLlmError::GgufError(format!(
            "unsupported GGUF version {version}; only v3 is supported"
        )));
    }
    let tensor_count = cur.read_u64()?;
    let metadata_kv_count = cur.read_u64()?;

    let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = cur.read_string()?;
        let value_type = cur.read_u32()?;
        let value = read_metadata_value(&mut cur, value_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()?;
        if n_dims > 8 {
            return Err(ScryLlmError::GgufError(format!(
                "tensor '{name}' has {n_dims} dims; refusing >8"
            )));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = cur.read_u64()?;
            shape.push(usize::try_from(d).map_err(|e| {
                ScryLlmError::GgufError(format!("tensor '{name}' dim overflow: {e}"))
            })?);
        }
        // GGUF stores dims in reverse vs the natural row-major order: a
        // shape recorded as `[d1, d0]` is the matrix `[d0, d1]`. Reverse
        // so callers see the natural row-major order.
        shape.reverse();
        let dtype = GgufDtype::from_ggml_type(cur.read_u32()?);
        let offset = cur.read_u64()?;
        tensors.insert(
            name.clone(),
            GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
            },
        );
    }

    // Tensor data section starts at the next multiple of `general.alignment`
    // (default 32). The alignment metadata key, if present, must be an
    // integer-typed value.
    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(32);
    let pos = cur.position();
    let pad = (alignment - (pos % alignment)) % alignment;
    let tensor_data_start = pos + pad;
    if tensor_data_start > bytes.len() {
        return Err(ScryLlmError::GgufError(format!(
            "tensor data start {tensor_data_start} past file end {}",
            bytes.len()
        )));
    }

    Ok(ParsedFile {
        tensor_data_start,
        metadata,
        tensors,
    })
}

fn read_metadata_value(cur: &mut Cursor<'_>, value_type: u32) -> Result<GgufMetadataValue> {
    match value_type {
        0 => Ok(GgufMetadataValue::U8(cur.read_u8()?)),
        1 => Ok(GgufMetadataValue::I8(cur.read_u8()? as i8)),
        2 => Ok(GgufMetadataValue::U16(cur.read_u16()?)),
        3 => Ok(GgufMetadataValue::I16(cur.read_u16()? as i16)),
        4 => Ok(GgufMetadataValue::U32(cur.read_u32()?)),
        5 => Ok(GgufMetadataValue::I32(cur.read_u32()? as i32)),
        6 => Ok(GgufMetadataValue::F32(f32::from_bits(cur.read_u32()?))),
        7 => Ok(GgufMetadataValue::Bool(cur.read_u8()? != 0)),
        8 => Ok(GgufMetadataValue::String(cur.read_string()?)),
        9 => {
            let elem_type = cur.read_u32()?;
            let n = cur.read_u64()?;
            let n_usize = usize::try_from(n)
                .map_err(|e| ScryLlmError::GgufError(format!("array length overflow: {e}")))?;
            let mut out = Vec::with_capacity(n_usize.min(1 << 20));
            for _ in 0..n_usize {
                out.push(read_metadata_value(cur, elem_type)?);
            }
            Ok(GgufMetadataValue::Array(out.into_boxed_slice()))
        }
        10 => Ok(GgufMetadataValue::U64(cur.read_u64()?)),
        11 => Ok(GgufMetadataValue::I64(cur.read_u64()? as i64)),
        12 => Ok(GgufMetadataValue::F64(f64::from_bits(cur.read_u64()?))),
        other => Err(ScryLlmError::GgufError(format!(
            "unknown metadata value type {other}"
        ))),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| ScryLlmError::GgufError(format!("read overflow at pos {}", self.pos)))?;
        if end > self.bytes.len() {
            return Err(ScryLlmError::GgufError(format!(
                "truncated at pos {}: need {n} bytes, have {}",
                self.pos,
                self.bytes.len() - self.pos
            )));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        let len = usize::try_from(len)
            .map_err(|e| ScryLlmError::GgufError(format!("string length overflow: {e}")))?;
        let raw = self.read_bytes(len)?;
        String::from_utf8(raw.to_vec()).map_err(|e| {
            ScryLlmError::GgufError(format!("non-utf8 string at pos {}: {e}", self.pos))
        })
    }
}
