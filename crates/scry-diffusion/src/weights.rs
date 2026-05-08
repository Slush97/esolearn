// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors weight loading and HF state-dict key mapping.
//!
//! HF Stable Diffusion checkpoints ship as safetensors files with PyTorch
//! state-dict naming conventions. Each subnet (UNet, VAE, text encoder)
//! lives in its own file when downloaded from `runwayml/stable-diffusion-v1-5`
//! / `stabilityai/stable-diffusion-xl-base-1.0` etc.:
//!
//! ```text
//! unet/diffusion_pytorch_model.safetensors
//! vae/diffusion_pytorch_model.safetensors
//! text_encoder/model.safetensors
//! tokenizer/{vocab.json, merges.txt}
//! ```
//!
//! [`SafetensorsCheckpoint`] is a thin owning wrapper around a memmapped
//! file: `open` mmaps + validates the safetensors header once, and
//! [`SafetensorsCheckpoint::tensors`] hands out a fresh `SafeTensors<'_>`
//! borrowed from that mmap on every call so model-loading code can use
//! the existing `scry_vision::checkpoint` helpers (`load_tensor`,
//! `load_conv2d`, etc.) directly. We rely on `scry-vision/safetensors`
//! being feature-chained from this crate's `safetensors` feature for the
//! cast helpers — no need to duplicate the bf16/f16/f32 byte conversions.

#[cfg(feature = "safetensors")]
use std::path::Path;

#[cfg(feature = "safetensors")]
use crate::error::{Error, Result};

/// A memory-mapped safetensors checkpoint.
///
/// Owns the mmap; the safetensors header is parsed eagerly at `open` for
/// fail-fast error reporting and re-parsed on each [`Self::tensors`] call
/// so callers can keep the returned view as long as they want without
/// fighting the borrow checker. The metadata block is small (a JSON header
/// at the start of the file), so the per-call parse cost is negligible
/// compared to the actual tensor reads.
#[cfg(feature = "safetensors")]
pub struct SafetensorsCheckpoint {
    mmap: memmap2::Mmap,
}

#[cfg(feature = "safetensors")]
impl SafetensorsCheckpoint {
    /// Memory-map a safetensors file and validate its header.
    ///
    /// # Errors
    /// Returns `Error::Llm` if the file is missing, unreadable, or fails
    /// to deserialize as a safetensors blob.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Llm(format!("open {}: {e}", path.display())))?;
        // SAFETY: standard pattern for read-only memmaps in this workspace
        // (mirrors scry-vision/src/models/{clip,resnet,sam}.rs). The
        // underlying file handle is dropped here, but the mmap remains
        // valid until `self` is dropped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Llm(format!("mmap {}: {e}", path.display())))?;
        // Parse once at open-time so callers see a malformed file before
        // they start asking for tensors.
        safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| Error::Llm(format!("parse {}: {e}", path.display())))?;
        Ok(Self { mmap })
    }

    /// Borrow a `SafeTensors` view rooted in the mmap. Cheap (re-parses
    /// the header but not the tensor bodies) — call once and reuse for
    /// the duration of a model load.
    ///
    /// # Errors
    /// Returns `Error::Llm` if the header has somehow become unparseable
    /// since `open` (shouldn't happen for a well-formed file).
    pub fn tensors(&self) -> Result<safetensors::SafeTensors<'_>> {
        safetensors::SafeTensors::deserialize(&self.mmap)
            .map_err(|e| Error::Llm(format!("re-parse safetensors header: {e}")))
    }

    /// All tensor names present in the checkpoint, in safetensors-internal
    /// order. Useful for `inspect_keys`-style debugging while wiring up
    /// `map_clip_text_keys` / `map_unet_keys` / `map_vae_keys`.
    ///
    /// # Errors
    /// Returns `Error::Llm` if the header parse fails (see `tensors`).
    pub fn names(&self) -> Result<Vec<String>> {
        Ok(self
            .tensors()?
            .names()
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect())
    }

    /// Read a tensor by name and copy it into a `Vec<f32>` (casting from
    /// `bf16` / `f16` if needed). Thin wrapper over
    /// `scry_vision::checkpoint::load_f32` so the rest of this crate
    /// doesn't need to import scry-vision directly.
    ///
    /// # Errors
    /// Returns `Error::Llm` if the tensor is missing or has an unsupported
    /// dtype.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensors = self.tensors()?;
        scry_vision::checkpoint::load_f32(&tensors, name)
            .map_err(|e| Error::Llm(format!("load tensor '{name}': {e}")))
    }
}

/// HF → scry-diffusion key mapping for the CLIP text encoder. M3 fills this
/// in once the encoder structs exist.
#[cfg(feature = "safetensors")]
pub fn map_clip_text_keys(_hf_key: &str) -> Option<String> {
    todo!("M3: translate HF text_model.encoder.layers.{{i}}.* -> our naming")
}

/// HF → scry-diffusion key mapping for the UNet. M5.
#[cfg(feature = "safetensors")]
pub fn map_unet_keys(_hf_key: &str) -> Option<String> {
    todo!(
        "M5: translate HF unet.down_blocks.{{i}}.resnets.{{j}}.* / .attentions.{{j}}.* / \
         .downsamplers.0.* -> our naming. Same for mid_block, up_blocks."
    )
}

/// HF → scry-diffusion key mapping for the VAE decoder. M4.
#[cfg(feature = "safetensors")]
pub fn map_vae_keys(_hf_key: &str) -> Option<String> {
    todo!("M4: translate HF decoder.up_blocks.{{i}}.* + post_quant_conv -> our naming")
}

#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Build a tiny safetensors blob with a few tensors of different
    /// dtypes (f32, bf16, f16) and write it to a temp file. Returns the
    /// `(tempdir, path)` so the file lives until the dir is dropped.
    fn write_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let f32_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = [1.0f32, -1.0, 0.5]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = [2.0f32, -2.0]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();

        let mut tensors: BTreeMap<&str, TensorView<'_>> = BTreeMap::new();
        tensors.insert(
            "alpha.f32",
            TensorView::new(Dtype::F32, vec![2, 2], &f32_bytes).unwrap(),
        );
        tensors.insert(
            "beta.bf16",
            TensorView::new(Dtype::BF16, vec![3], &bf16_bytes).unwrap(),
        );
        tensors.insert(
            "gamma.f16",
            TensorView::new(Dtype::F16, vec![2], &f16_bytes).unwrap(),
        );

        let blob = safetensors::serialize(&tensors, &None).expect("serialize fixture");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fixture.safetensors");
        std::fs::write(&path, &blob).expect("write fixture");
        (dir, path)
    }

    #[test]
    fn open_and_list_names() {
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).expect("open");
        let mut names = ckpt.names().expect("names");
        names.sort();
        assert_eq!(names, vec!["alpha.f32", "beta.bf16", "gamma.f16"]);
    }

    #[test]
    fn tensor_f32_reads_f32_directly() {
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).unwrap();
        let v = ckpt.tensor_f32("alpha.f32").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_f32_casts_bf16_to_f32() {
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).unwrap();
        let v = ckpt.tensor_f32("beta.bf16").unwrap();
        // bf16 keeps the f32 exponent and 7 mantissa bits, so simple
        // values like 1.0, -1.0, 0.5 round-trip exactly.
        assert_eq!(v, vec![1.0, -1.0, 0.5]);
    }

    #[test]
    fn tensor_f32_casts_f16_to_f32() {
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).unwrap();
        let v = ckpt.tensor_f32("gamma.f16").unwrap();
        // f16 has plenty of precision for 2.0 / -2.0.
        assert_eq!(v, vec![2.0, -2.0]);
    }

    #[test]
    fn tensor_f32_missing_key_returns_error() {
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).unwrap();
        assert!(ckpt.tensor_f32("does.not.exist").is_err());
    }

    #[test]
    fn open_missing_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.safetensors");
        assert!(SafetensorsCheckpoint::open(&path).is_err());
    }

    #[test]
    fn open_malformed_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.safetensors");
        std::fs::write(&path, b"this is not safetensors").unwrap();
        assert!(SafetensorsCheckpoint::open(&path).is_err());
    }

    #[test]
    fn tensors_view_compatible_with_scry_vision_helpers() {
        // The whole point of mirroring scry-vision is that callers can
        // reach for the existing `load_tensor` / `load_conv2d` etc.
        // helpers. Smoke-test the integration by going through
        // `load_f32` directly on the borrowed view.
        let (_dir, path) = write_fixture();
        let ckpt = SafetensorsCheckpoint::open(&path).unwrap();
        let view = ckpt.tensors().unwrap();
        let v = scry_vision::checkpoint::load_f32(&view, "alpha.f32").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
