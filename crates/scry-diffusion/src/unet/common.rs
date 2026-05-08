// SPDX-License-Identifier: MIT OR Apache-2.0
//! Helpers shared across `unet/{resblock, attention, blocks, mod}.rs`:
//! [`GroupNormParams`] holding affine weights for `MathBackend::group_norm`,
//! and HF-`[out, in]` → scry-llm-`[in, out]` linear loaders that handle the
//! transpose-on-load while consuming the safetensors keys for the 100%
//! consumption check. Kept private to the crate because the UNet is the
//! only caller for these specific shapes (the VAE decoder rolls its own
//! local transpose helpers).

#[cfg(feature = "safetensors")]
use std::collections::HashSet;

use scry_llm::backend::MathBackend;
#[cfg(feature = "safetensors")]
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

#[cfg(feature = "safetensors")]
use crate::error::{Error, Result};

/// 1D GroupNorm parameter pack. Compute goes through
/// `MathBackend::group_norm`; this struct just holds the affine weights
/// and shape metadata so callers don't have to plumb `num_groups` /
/// `channels` separately.
pub(crate) struct GroupNormParams<B: MathBackend> {
    pub(crate) weight: Tensor<B>,
    pub(crate) bias: Tensor<B>,
    pub(crate) num_groups: usize,
    pub(crate) channels: usize,
}

/// Load `{prefix}.weight` + `{prefix}.bias` as a GroupNorm affine pair and
/// register both keys for the consumption check.
#[cfg(feature = "safetensors")]
pub(crate) fn load_group_norm<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<GroupNormParams<B>> {
    use scry_vision::checkpoint::load_tensor;

    let w_key = format!("{prefix}.weight");
    let b_key = format!("{prefix}.bias");
    let weight = load_tensor::<B>(view, &w_key, &[channels])
        .map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
    let bias = load_tensor::<B>(view, &b_key, &[channels])
        .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
    consume(&w_key);
    consume(&b_key);
    Ok(GroupNormParams {
        weight,
        bias,
        num_groups,
        channels,
    })
}

/// Load a HF Linear (`{prefix}.weight: [out, in]`, optional `bias: [out]`)
/// into scry-llm's `[in, out]` weight layout, transposing the weight
/// once at load time. `has_bias = false` is used by the SD attention
/// `to_q` / `to_k` / `to_v` projections (HF `nn.Linear(bias=False)`).
///
/// Returns `(weight [in, out], bias_or_zeros [out])`. When `has_bias`
/// is true, both `{prefix}.weight` and `{prefix}.bias` are consumed;
/// when false, only `{prefix}.weight` is consumed and the bias tensor
/// is filled with zeros so downstream `matmul_bias` calls can stay
/// unconditional.
#[cfg(feature = "safetensors")]
pub(crate) fn load_linear<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    consume: &mut impl FnMut(&str),
) -> Result<(Tensor<B>, Tensor<B>)> {
    use scry_vision::checkpoint::{load_f32, load_tensor};

    let w_key = format!("{prefix}.weight");
    let raw = load_f32(view, &w_key).map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
    let expected = out_features * in_features;
    if raw.len() != expected {
        return Err(Error::Llm(format!(
            "{w_key}: expected {expected} elements ({out_features}×{in_features}), got {}",
            raw.len()
        )));
    }
    let mut transposed = vec![0.0f32; expected];
    for in_i in 0..in_features {
        for out_i in 0..out_features {
            transposed[in_i * out_features + out_i] = raw[out_i * in_features + in_i];
        }
    }
    let weight = Tensor::from_vec(transposed, Shape::new(&[in_features, out_features]));
    consume(&w_key);

    let bias = if has_bias {
        let b_key = format!("{prefix}.bias");
        let b = load_tensor::<B>(view, &b_key, &[out_features])
            .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
        consume(&b_key);
        b
    } else {
        Tensor::from_vec(vec![0.0; out_features], Shape::new(&[out_features]))
    };
    Ok((weight, bias))
}

/// Sort + truncate a missing-key set into a `cargo test`-readable error.
/// Used at the tail of every from_safetensors loader.
#[cfg(feature = "safetensors")]
pub(crate) fn missing_keys_error(
    label: &str,
    relevant: &HashSet<String>,
    consumed: &HashSet<String>,
) -> Option<Error> {
    let missing: Vec<String> = relevant.difference(consumed).cloned().collect();
    if missing.is_empty() {
        return None;
    }
    let mut sorted = missing;
    sorted.sort();
    let head = sorted
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let tail = if sorted.len() > 8 { ", ..." } else { "" };
    Some(Error::Llm(format!(
        "{label}: {} keys not consumed by loader: {head}{tail}",
        sorted.len()
    )))
}
