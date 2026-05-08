// SPDX-License-Identifier: MIT OR Apache-2.0
//! Smoke tests that exercise the M2 primitives (tokenizer +
//! `SafetensorsCheckpoint`) against a real Stable Diffusion 1.5 snapshot
//! at `crates/scry-diffusion/.assets/sd-1-5/`.
//!
//! Each test no-ops with a stderr hint if the snapshot isn't populated
//! locally, so the suite is safe to run on a clean checkout. Populate the
//! snapshot per `crates/scry-diffusion/ASSETS.md`.

#![cfg(feature = "safetensors")]

use std::path::{Path, PathBuf};

use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::vae::{decoder::VaeDecoderConfig, VaeDecoder};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_llm::backend::cpu::CpuBackend;

fn snapshot_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5")
}

fn skip_if_missing(path: &Path, label: &str) -> bool {
    if path.exists() {
        return false;
    }
    eprintln!(
        "SKIP {label}: {} not found — populate per crates/scry-diffusion/ASSETS.md",
        path.display()
    );
    true
}

#[test]
fn tokenizer_loads_real_clip_vocab() {
    let dir = snapshot_root().join("tokenizer");
    if skip_if_missing(&dir.join("vocab.json"), "tokenizer_loads_real_clip_vocab") {
        return;
    }
    let tok = Tokenizer::from_dir(&dir).expect("from_dir");
    // CLIP / OpenCLIP vocab is 49_408 (49_406 BPE pieces + 2 specials).
    assert_eq!(tok.vocab_size(), 49_408, "unexpected CLIP vocab size");

    // Empty prompt: just BOS + EOS framing. With nothing in between,
    // every middle slot is the EOS pad.
    let ids = tok.encode("").expect("encode empty");
    assert_eq!(ids.len(), Tokenizer::MAX_SEQ_LEN);
    assert_eq!(ids[0], Tokenizer::BOS_TOKEN);
    assert!(ids[1..].iter().all(|&id| id == Tokenizer::EOS_TOKEN));

    // Non-empty prompt: BOS, some real ids, then EOS, then EOS pad.
    let ids = tok.encode("a photo of a cat").expect("encode non-empty");
    assert_eq!(ids.len(), Tokenizer::MAX_SEQ_LEN);
    assert_eq!(ids[0], Tokenizer::BOS_TOKEN);
    // Whatever the BPE produces, the first non-BOS slot should not be
    // EOS — there's actual content here.
    assert_ne!(ids[1], Tokenizer::EOS_TOKEN);
    // And the trailing tail must still be EOS pad.
    assert_eq!(ids[Tokenizer::MAX_SEQ_LEN - 1], Tokenizer::EOS_TOKEN);
}

#[test]
fn checkpoint_opens_text_encoder() {
    let path = snapshot_root().join("text_encoder/model.safetensors");
    if skip_if_missing(&path, "checkpoint_opens_text_encoder") {
        return;
    }
    let ckpt = SafetensorsCheckpoint::open(&path).expect("open text_encoder");
    let names = ckpt.names().expect("names");
    assert!(!names.is_empty(), "text_encoder has zero tensors?");
    // CLIP-L's token embedding key is well-known; if this is missing, the
    // checkpoint was downloaded into the wrong directory.
    assert!(
        names
            .iter()
            .any(|n| n == "text_model.embeddings.token_embedding.weight"),
        "missing CLIP token_embedding.weight in text_encoder/model.safetensors; \
         got {} tensors, e.g. {:?}",
        names.len(),
        names.iter().take(3).collect::<Vec<_>>()
    );
    // And one tensor read should round-trip through tensor_f32 cleanly.
    let _emb = ckpt
        .tensor_f32("text_model.embeddings.token_embedding.weight")
        .expect("tensor_f32 on token_embedding");
}

#[test]
fn checkpoint_opens_vae() {
    let path = snapshot_root().join("vae/diffusion_pytorch_model.safetensors");
    if skip_if_missing(&path, "checkpoint_opens_vae") {
        return;
    }
    let ckpt = SafetensorsCheckpoint::open(&path).expect("open vae");
    let names = ckpt.names().expect("names");
    assert!(
        names.iter().any(|n| n.starts_with("decoder.")),
        "vae checkpoint has no `decoder.*` tensors — wrong file?"
    );
}

#[test]
fn clip_text_encoder_loads_all_keys() {
    let path = snapshot_root().join("text_encoder/model.safetensors");
    if skip_if_missing(&path, "clip_text_encoder_loads_all_keys") {
        return;
    }
    let ckpt = SafetensorsCheckpoint::open(&path).expect("open");
    let encoder =
        ClipTextEncoder::<CpuBackend>::from_safetensors(ClipTextConfig::clip_vit_l(), &ckpt)
            .expect("from_safetensors");
    assert_eq!(encoder.d_model(), 768);
    assert_eq!(encoder.num_layers(), 12);
}

#[test]
fn vae_decoder_loads_all_keys() {
    let path = snapshot_root().join("vae/diffusion_pytorch_model.safetensors");
    if skip_if_missing(&path, "vae_decoder_loads_all_keys") {
        return;
    }
    let ckpt = SafetensorsCheckpoint::open(&path).expect("open vae");
    let _decoder = VaeDecoder::<CpuBackend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &ckpt)
        .expect("from_safetensors");
}

#[test]
fn checkpoint_opens_unet() {
    let path = snapshot_root().join("unet/diffusion_pytorch_model.safetensors");
    if skip_if_missing(&path, "checkpoint_opens_unet") {
        return;
    }
    let ckpt = SafetensorsCheckpoint::open(&path).expect("open unet");
    let names = ckpt.names().expect("names");
    assert!(
        names.iter().any(|n| n.starts_with("down_blocks.")),
        "unet checkpoint has no `down_blocks.*` tensors — wrong file?"
    );
}
