// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-image hash gate for txt2img.
//!
//! Pinned tiny config: SD 1.5, DDIM, prompt="a photo of a cat",
//! seed=42, cfg=7.5, 4 steps, 64×64, bf16 matmul ON.
//!
//! The committed hash in `tests/fixtures/golden_hash.txt` is the
//! SHA-256 of the post-decode image after quantising to u8 (HWC). The
//! u8 quantisation absorbs sub-LSB numerical drift from cuBLAS/cuDNN
//! while still failing on any meaningful pipeline change.
//!
//! Compile-time gates: `safetensors` + `scry-gpu-cuda`. CI doesn't have
//! the GPU and doesn't ship weights, so this test never compiles in the
//! workspace CI matrix; the pre-push hook runs it locally with the
//! GPU feature set explicitly enabled.
//!
//! Update the fixture hash deliberately (after a parity-affecting
//! change) by setting `GOLDEN_HASH_UPDATE=1` and re-running the test —
//! it overwrites the fixture and passes once.

#![cfg(all(feature = "safetensors", feature = "scry-gpu-cuda"))]

use std::path::{Path, PathBuf};

use scry_diffusion::scheduler::ddim::{DdimConfig, DdimScheduler};
use scry_diffusion::text_encoder::clip_text::{ClipTextConfig, ClipTextEncoder};
use scry_diffusion::tokenizer::Tokenizer;
use scry_diffusion::unet::{Unet, UnetConfig};
use scry_diffusion::vae::{decoder::VaeDecoderConfig, VaeDecoder};
use scry_diffusion::weights::SafetensorsCheckpoint;
use scry_diffusion::{GenerationParams, SdPipeline};
use scry_llm::backend::scry_gpu::ScryGpuBackend as Backend;
use sha2::{Digest, Sha256};

const PROMPT: &str = "a photo of a cat";
const NEGATIVE: &str = "";
const STEPS: u32 = 4;
const CFG: f32 = 7.5;
const SEED: u64 = 42;
const SIZE: u32 = 64;

fn snapshot_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".assets/sd-1-5")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_hash.txt")
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
fn txt2img_golden_hash() {
    let snapshot = snapshot_root();
    if skip_if_missing(
        &snapshot.join("tokenizer/vocab.json"),
        "txt2img_golden_hash",
    ) {
        return;
    }
    if skip_if_missing(
        &snapshot.join("text_encoder/model.safetensors"),
        "txt2img_golden_hash",
    ) {
        return;
    }
    if skip_if_missing(
        &snapshot.join("unet/diffusion_pytorch_model.safetensors"),
        "txt2img_golden_hash",
    ) {
        return;
    }
    if skip_if_missing(
        &snapshot.join("vae/diffusion_pytorch_model.safetensors"),
        "txt2img_golden_hash",
    ) {
        return;
    }

    // Pin backend toggles to the gate config. golden_hash is the only
    // test in this binary, so process-global state is fine to set once.
    #[cfg(feature = "scry-gpu-bf16")]
    Backend::set_bf16_matmul(true).expect("set_bf16_matmul");
    #[cfg(feature = "scry-gpu-cudnn")]
    Backend::set_cudnn_conv(true).expect("set_cudnn_conv");

    let tokenizer = Tokenizer::from_dir(snapshot.join("tokenizer")).expect("tokenizer");
    let text_ckpt =
        SafetensorsCheckpoint::open(snapshot.join("text_encoder/model.safetensors")).expect("clip");
    let text_encoder =
        ClipTextEncoder::<Backend>::from_safetensors(ClipTextConfig::clip_vit_l(), &text_ckpt)
            .expect("clip from_safetensors");
    let unet_ckpt =
        SafetensorsCheckpoint::open(snapshot.join("unet/diffusion_pytorch_model.safetensors"))
            .expect("unet");
    let unet =
        Unet::<Backend>::from_safetensors(UnetConfig::sd_1_5(), &unet_ckpt).expect("unet build");
    let vae_ckpt =
        SafetensorsCheckpoint::open(snapshot.join("vae/diffusion_pytorch_model.safetensors"))
            .expect("vae");
    let vae = VaeDecoder::<Backend>::from_safetensors(VaeDecoderConfig::sd_1_5(), &vae_ckpt)
        .expect("vae build");
    let scheduler = DdimScheduler::new(DdimConfig::sd_1_5()).expect("ddim");

    let mut pipeline = SdPipeline {
        tokenizer,
        text_encoder,
        unet,
        vae,
        vae_encoder: None,
        scheduler,
        progress: None,
    };
    pipeline.to_device();

    let img = pipeline
        .generate(&GenerationParams {
            prompt: PROMPT.into(),
            negative_prompt: NEGATIVE.into(),
            num_inference_steps: STEPS,
            guidance_scale: CFG,
            seed: SEED,
            size: (SIZE, SIZE),
        })
        .expect("generate");

    let dims = img.shape.dims().to_vec();
    let (channels, h, w) = match dims.as_slice() {
        [c, h, w] => (*c, *h, *w),
        other => panic!("expected [C,H,W], got {other:?}"),
    };
    assert_eq!(channels, 3, "expected RGB output");
    let pixels = img.to_vec(); // [3, H, W] in [0, 1]

    // Quantise to u8 HWC — same packing the txt2img example uses to
    // produce the PNG, so a hash mismatch here corresponds to a
    // mismatch a user would visually notice. The u8 round absorbs
    // sub-LSB numerical drift in cuBLAS/cuDNN.
    let plane = h * w;
    let mut hwc = vec![0_u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let src_off = y * w + x;
            let dst = (y * w + x) * 3;
            for c in 0..3 {
                let v = pixels[c * plane + src_off].clamp(0.0, 1.0);
                hwc[dst + c] = (v * 255.0 + 0.5) as u8;
            }
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(&hwc);
    let actual = format!("{:x}", hasher.finalize());

    let fixture = fixture_path();
    let update_mode = std::env::var("GOLDEN_HASH_UPDATE").is_ok_and(|v| !v.is_empty() && v != "0");

    if update_mode || !fixture.exists() {
        if let Some(parent) = fixture.parent() {
            std::fs::create_dir_all(parent).expect("mkdir fixtures");
        }
        std::fs::write(&fixture, format!("{actual}\n")).expect("write fixture");
        eprintln!(
            "[golden-hash] {} fixture {} -> {actual}",
            if update_mode { "updated" } else { "seeded" },
            fixture.display(),
        );
        return;
    }

    let expected = std::fs::read_to_string(&fixture)
        .expect("read fixture")
        .trim()
        .to_string();
    assert_eq!(
        actual,
        expected,
        "golden image hash drifted at config size={SIZE} steps={STEPS} seed={SEED} \
         prompt={PROMPT:?} cfg={CFG}.\n\
         expected: {expected}\n\
         actual:   {actual}\n\
         If this is intentional, re-run with GOLDEN_HASH_UPDATE=1 and commit \
         {}.",
        fixture.display(),
    );
}
