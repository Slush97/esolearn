// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration smoke test for the GGUF parser against real-world files.
//!
//! Drives a path via `SCRY_LLM_GGUF_PATH`; falls back to a known fixture
//! path on the dev machine. `#[ignore]` so CI doesn't depend on the file.

#![cfg(feature = "gguf")]

use std::path::PathBuf;

use scry_llm::gguf::{GgufDtype, GgufFile};

fn locate_gguf() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SCRY_LLM_GGUF_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    // Dev-machine fallback. Skip silently if absent.
    let dev = PathBuf::from(
        "/home/esoc/.lmstudio/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/\
         Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
    );
    if dev.exists() {
        Some(dev)
    } else {
        None
    }
}

#[test]
#[ignore = "requires a GGUF model on disk; set SCRY_LLM_GGUF_PATH"]
fn parses_real_gguf() {
    let Some(path) = locate_gguf() else {
        eprintln!("no GGUF available; set SCRY_LLM_GGUF_PATH");
        return;
    };

    let file = GgufFile::open(&path).expect("parse GGUF");

    let arch = file.metadata_string("general.architecture");
    eprintln!("general.architecture = {arch:?}");
    assert!(arch.is_some(), "every GGUF must have general.architecture");

    let n_tensors = file.tensor_names().count();
    eprintln!("tensor count = {n_tensors}");
    assert!(n_tensors > 0);

    // Print the first 10 tensor names + shapes for eyeball.
    let mut tensors: Vec<_> = file.tensor_names().collect();
    tensors.sort_unstable();
    for name in tensors.iter().take(10) {
        let info = file.tensor_info(name).unwrap();
        eprintln!(
            "  {name:60}  shape={:?}  dtype={:?}",
            info.shape, info.dtype
        );
    }

    // Pick one tensor per supported dtype and confirm dequant works end-to-end.
    let probe_dtypes = [GgufDtype::F32, GgufDtype::Q4K, GgufDtype::Q8_0];
    for dtype in probe_dtypes {
        let Some(name) = tensors
            .iter()
            .find(|n| file.tensor_info(n).unwrap().dtype == dtype)
            .copied()
        else {
            eprintln!("(no {dtype:?} tensor in this file)");
            continue;
        };
        let info = file.tensor_info(name).unwrap();
        let n: usize = info.shape.iter().product();
        let data = file.tensor_f32(name).expect("dequant");
        let preview = &data[..5.min(data.len())];
        eprintln!("dequant {name}: dtype={dtype:?} n_elems={n} first={preview:?}");
        assert_eq!(data.len(), n);
        let finite_nonzero = data.iter().any(|&v| v != 0.0 && v.is_finite());
        assert!(finite_nonzero, "tensor data looks degenerate");
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "found {nan_count} NaNs in {name}");
    }
}
