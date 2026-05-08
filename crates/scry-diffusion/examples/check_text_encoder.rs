// SPDX-License-Identifier: MIT OR Apache-2.0
//! M3 numerical-equivalence gate for the CLIP text encoder.
//!
//! Loads our `ClipTextEncoder` from the SD 1.5 checkpoint, encodes a
//! fixed prompt, and byte-compares the output against a reference dump
//! from HF `transformers.CLIPTextModel`. The reference dump is produced
//! by `python/dump_clip_text_ref.py`. Tolerance is 1e-4 absolute, the
//! HANDOFF gate for "this transformer matches HF in fp32".
//!
//! Usage:
//!
//! ```text
//! # Once: produce the reference (via the venv shared with scry-vision):
//! crates/scry-vision/.venv/bin/python \
//!     crates/scry-diffusion/python/dump_clip_text_ref.py
//!
//! # Then: run the gate.
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_text_encoder
//!
//! # Or with a custom prompt (must be regenerated on the Python side
//! # with the same --prompt to match):
//! cargo run -p scry-diffusion --release \
//!     --features safetensors --example check_text_encoder -- "a red apple"
//! ```
//!
//! The example also asserts our DIY CLIP BPE tokenizer agrees with HF's
//! `CLIPTokenizer` on the input ids before running the forward — a
//! tokenizer disagreement on `[2..76]` would otherwise look like a
//! catastrophic encoder failure, not a tokenizer one.

use std::path::PathBuf;
use std::time::Instant;

use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;

const TOL_ABS: f32 = 1e-4;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a photo of a cat".to_string());

    let snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5");
    let refs_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/refs");
    let ref_path = refs_dir.join(format!("clip_text_{}.safetensors", slug(&prompt)));

    if !ref_path.exists() {
        return Err(format!(
            "reference dump not found at {}\n\
             generate it first:\n  \
             crates/scry-vision/.venv/bin/python \
             crates/scry-diffusion/python/dump_clip_text_ref.py --prompt {prompt:?}",
            ref_path.display()
        )
        .into());
    }

    // ---- Step 1: tokenizer agreement ----------------------------------
    let tok = Tokenizer::from_dir(snapshot.join("tokenizer"))?;
    let our_ids = tok.encode(&prompt)?;
    let ref_ckpt = SafetensorsCheckpoint::open(&ref_path)?;
    let hf_ids = read_int64_tensor(&ref_ckpt, "input_ids")?;

    if hf_ids.len() != our_ids.len() {
        return Err(format!(
            "tokenizer length mismatch: ours {} vs HF {}",
            our_ids.len(),
            hf_ids.len()
        )
        .into());
    }
    let mismatches: Vec<(usize, u32, u32)> = our_ids
        .iter()
        .zip(hf_ids.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, *a, *b))
        .collect();

    println!("prompt:       {prompt:?}");
    println!("first 8 ids:  {:?}", &our_ids[..8.min(our_ids.len())]);
    if mismatches.is_empty() {
        println!(
            "✓ tokenizer matches HF CLIPTokenizer on all {} ids",
            our_ids.len()
        );
    } else {
        println!(
            "✗ tokenizer disagrees with HF at {} positions:",
            mismatches.len()
        );
        for (i, ours, theirs) in mismatches.iter().take(10) {
            println!("    pos {i}: ours={ours} hf={theirs}");
        }
        return Err("tokenizer mismatch".into());
    }

    // ---- Step 2: load the encoder ------------------------------------
    println!("\nloading text encoder weights...");
    let model_path = snapshot.join("text_encoder/model.safetensors");
    let model_ckpt = SafetensorsCheckpoint::open(&model_path)?;
    let t0 = Instant::now();
    let encoder =
        ClipTextEncoder::<CpuBackend>::from_safetensors(ClipTextConfig::clip_vit_l(), &model_ckpt)?;
    println!(
        "  loaded {} layers, d_model={}, in {:.1}s",
        encoder.num_layers(),
        encoder.d_model(),
        t0.elapsed().as_secs_f32()
    );

    // ---- Step 3: forward pass ----------------------------------------
    println!("running forward pass on CPU...");
    let t0 = Instant::now();
    let our_out = encoder.forward(&our_ids)?;
    let our_vec = our_out.to_vec();
    println!(
        "  forward took {:.1}s, output shape {:?}, {} elements",
        t0.elapsed().as_secs_f32(),
        our_out.shape.dims(),
        our_vec.len()
    );

    // ---- Step 4: compare to HF reference ------------------------------
    let hf_vec = ref_ckpt.tensor_f32("last_hidden_state")?;
    if our_vec.len() != hf_vec.len() {
        return Err(format!(
            "output length mismatch: ours {} vs HF {}",
            our_vec.len(),
            hf_vec.len()
        )
        .into());
    }

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_pos = 0usize;
    for (i, (a, b)) in our_vec.iter().zip(hf_vec.iter()).enumerate() {
        let d = (a - b).abs();
        sum_diff += f64::from(d);
        if d > max_diff {
            max_diff = d;
            max_pos = i;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_diff = sum_diff / our_vec.len() as f64;
    let (tok_pos, dim_pos) = (max_pos / 768, max_pos % 768);

    println!("\ndiff vs HF reference:");
    println!("  max abs diff:  {max_diff:.3e} at token {tok_pos} dim {dim_pos}");
    println!("  mean abs diff: {mean_diff:.3e}");
    println!("  tolerance:     {TOL_ABS:.0e}");

    if max_diff > TOL_ABS {
        return Err(format!(
            "max diff {max_diff:.3e} exceeds tolerance {TOL_ABS:.0e} — encoder fails M3 gate"
        )
        .into());
    }
    println!("\n✓ M3 numerical-equivalence gate PASSED");
    Ok(())
}

/// Lowercase + non-alphanumeric → `_`, trimmed; mirrors the slug rule
/// in `dump_clip_text_ref.py::default_out_path`.
fn slug(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while out.starts_with('_') {
        out.remove(0);
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

/// Read an `i64` safetensors tensor as a `Vec<u32>`. CLIP token ids fit
/// in 31 bits, so the cast is safe; a real out-of-range value would
/// indicate corruption and we'd rather panic via `try_into` than wrap.
fn read_int64_tensor(
    ckpt: &SafetensorsCheckpoint,
    name: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let view = ckpt.tensors()?;
    let t = view.tensor(name)?;
    let mut out = Vec::with_capacity(t.data().len() / 8);
    for chunk in t.data().chunks_exact(8) {
        let raw: [u8; 8] = chunk.try_into().unwrap();
        let v = i64::from_le_bytes(raw);
        u32::try_from(v).map(|u| out.push(u))?;
    }
    Ok(out)
}
