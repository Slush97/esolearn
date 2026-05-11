// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden-tensor validation for the GGUF dequant path.
//!
//! Compares our `GgufFile::tensor_f32` output against fixtures produced
//! by llama.cpp's canonical `gguf` Python package
//! (see `python/dump_gguf_golden.py`). Two fixture sets:
//!
//! - `gguf_golden/` — Gemma-4-E4B Q4_K_M sample (covers F32, Q4_K)
//! - `gguf_golden_llama/` — Llama 3.2 1B Q4_K_M sample (covers F32, Q4_K, Q6_K)
//!
//! Each test locates its source GGUF via an env var fallback, then a
//! dev-machine fallback. `#[ignore]`-gated since CI doesn't have the
//! source GGUFs on disk.

#![cfg(feature = "gguf")]

use std::path::{Path, PathBuf};

use scry_llm::gguf::GgufFile;

fn fixture_dir(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(sub)
}

fn check_path(p: PathBuf) -> Option<PathBuf> {
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn locate_gemma_gguf() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SCRY_LLM_GGUF_PATH") {
        if let Some(p) = check_path(PathBuf::from(p)) {
            return Some(p);
        }
    }
    check_path(PathBuf::from(
        "/home/esoc/.lmstudio/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/\
         Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
    ))
}

fn locate_llama_gguf() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SCRY_LLM_LLAMA_GGUF_PATH") {
        if let Some(p) = check_path(PathBuf::from(p)) {
            return Some(p);
        }
    }
    check_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    )
}

fn load_golden_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden bin");
    assert_eq!(
        bytes.len() % 4,
        0,
        "golden bin {} not f32-aligned",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare every tensor named in the fixture's `manifest.json` between
/// our `GgufFile::tensor_f32` and the gguf-py golden binary. Strict
/// `max_abs_diff == 0` since both implementations apply the same fixed
/// sequence of f32 ops per element.
fn assert_dequant_matches_golden(source_gguf: &Path, fixtures: &Path) {
    let manifest_path = fixtures.join("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read manifest"))
            .expect("parse manifest");
    let file = GgufFile::open(source_gguf).expect("parse gguf");

    let entries = manifest["tensors"].as_array().expect("tensors array");
    assert!(!entries.is_empty(), "manifest has no tensors");
    eprintln!("source: {}", source_gguf.display());

    for entry in entries {
        let name = entry["name"].as_str().expect("name");
        let head_count = entry["head_count"].as_u64().expect("head_count") as usize;
        let bin = entry["bin"].as_str().expect("bin");
        let source_dtype = entry["source_dtype"].as_str().expect("source_dtype");

        let golden = load_golden_f32(&fixtures.join(bin));
        assert_eq!(golden.len(), head_count, "{name}: golden length");

        let ours = file
            .tensor_f32(name)
            .unwrap_or_else(|e| panic!("{name}: dequant via scry-llm failed: {e}"));
        assert!(
            ours.len() >= head_count,
            "{name}: our dequant produced only {} elements, need {head_count}",
            ours.len()
        );
        let head = &ours[..head_count];

        let mut max_diff: f32 = 0.0;
        let mut first_mismatch: Option<(usize, f32, f32)> = None;
        for (i, (&a, &b)) in head.iter().zip(golden.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            if a.to_bits() != b.to_bits() && first_mismatch.is_none() {
                first_mismatch = Some((i, a, b));
            }
        }
        eprintln!("{name:<48} {source_dtype:<6} head={head_count:>4} max_abs_diff={max_diff:e}");
        // First few values from both sides for eyeball/debug
        eprintln!("  ours  [0..8] = {:?}", &head[..8]);
        eprintln!("  golden[0..8] = {:?}", &golden[..8]);
        if let Some((i, a, b)) = first_mismatch {
            eprintln!(
                "  first bitwise mismatch at i={i}: ours=0x{:08x} ({a}) gguf-py=0x{:08x} ({b})",
                a.to_bits(),
                b.to_bits()
            );
        }
        assert!(
            max_diff == 0.0,
            "{name}: max_abs_diff {max_diff:e} != 0 (golden is gguf-py)"
        );
    }
}

#[test]
#[ignore = "requires Gemma source GGUF; set SCRY_LLM_GGUF_PATH or use dev fallback"]
fn gemma_q4km_matches_gguf_py_dequant() {
    let Some(path) = locate_gemma_gguf() else {
        eprintln!("skipping: no Gemma GGUF found");
        return;
    };
    assert_dequant_matches_golden(&path, &fixture_dir("gguf_golden"));
}

#[test]
#[ignore = "requires Llama source GGUF; set SCRY_LLM_LLAMA_GGUF_PATH or download to tests/fixtures/"]
fn llama_3_2_1b_q4km_matches_gguf_py_dequant() {
    let Some(path) = locate_llama_gguf() else {
        eprintln!("skipping: no Llama GGUF found");
        return;
    };
    assert_dequant_matches_golden(&path, &fixture_dir("gguf_golden_llama"));
}
