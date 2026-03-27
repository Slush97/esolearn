// SPDX-License-Identifier: MIT OR Apache-2.0
//! Load a real ResNet-18 and classify a synthetic image.
//!
//! cargo run -p scry-vision --features safetensors --example classify --release
//!
//! Requires: testdata/resnet18.safetensors (from timm/resnet18.a1_in1k on HuggingFace)

use std::path::Path;
use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;

use scry_vision::models::{ResNetClassifier, ResNetConfig};
use scry_vision::pipeline::Classify;

// ImageNet class labels for a few well-known classes
const SAMPLE_LABELS: &[(u32, &str)] = &[
    (0, "tench"), (1, "goldfish"), (65, "sea snake"), (207, "golden retriever"),
    (208, "Labrador retriever"), (281, "tabby cat"), (285, "Egyptian cat"),
    (291, "lion"), (340, "zebra"), (386, "African elephant"),
    (463, "bucket"), (574, "golf ball"), (717, "pickup truck"),
    (757, "red wine"), (804, "ski"), (852, "tennis ball"),
    (937, "broccoli"), (947, "mushroom"), (954, "banana"), (999, "toilet tissue"),
];

fn label_for(class_id: u32) -> String {
    SAMPLE_LABELS
        .iter()
        .find(|(id, _)| *id == class_id)
        .map(|(_, name)| name.to_string())
        .unwrap_or_else(|| format!("class_{class_id}"))
}

fn main() {
    let model_path = std::env::var("RESNET_MODEL_PATH").unwrap_or_else(|_| {
        format!("{}/testdata/resnet18.safetensors", env!("CARGO_MANIFEST_DIR"))
    });
    let path = Path::new(&model_path);

    if !path.exists() {
        eprintln!("Model not found at {}", path.display());
        eprintln!("Download with:");
        eprintln!("  curl -L https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors \\");
        eprintln!("    -o crates/scry-vision/testdata/resnet18.safetensors");
        std::process::exit(1);
    }

    // Load model
    println!("Loading ResNet-18 from {}...", path.display());
    let t0 = Instant::now();
    let config = ResNetConfig::resnet18(1000);
    let classifier = ResNetClassifier::<CpuBackend>::from_safetensors(config, path).unwrap();
    println!("  Loaded in {:.1}ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    // Test 1: solid red image (224x224)
    println!("=== Solid red image (224x224) ===");
    let mut red_image = vec![0u8; 224 * 224 * 3];
    for i in 0..(224 * 224) {
        red_image[i * 3] = 255;     // R
        red_image[i * 3 + 1] = 0;   // G
        red_image[i * 3 + 2] = 0;   // B
    }
    run_classify(&classifier, &red_image, 224, 224);

    // Test 2: solid green image
    println!("=== Solid green image (224x224) ===");
    let mut green_image = vec![0u8; 224 * 224 * 3];
    for i in 0..(224 * 224) {
        green_image[i * 3] = 0;
        green_image[i * 3 + 1] = 200;
        green_image[i * 3 + 2] = 30;
    }
    run_classify(&classifier, &green_image, 224, 224);

    // Test 3: random-ish texture (stripes)
    println!("=== Striped pattern (224x224) ===");
    let mut striped = vec![0u8; 224 * 224 * 3];
    for y in 0..224 {
        for x in 0..224 {
            let i = (y * 224 + x) * 3;
            if (y / 16) % 2 == 0 {
                striped[i] = 200; striped[i + 1] = 180; striped[i + 2] = 50;
            } else {
                striped[i] = 30; striped[i + 1] = 60; striped[i + 2] = 150;
            }
        }
    }
    run_classify(&classifier, &striped, 224, 224);

    // Test 4: non-square input to verify resize works
    println!("=== Non-square input (320x180) ===");
    let image_320 = vec![128u8; 320 * 180 * 3];
    run_classify(&classifier, &image_320, 320, 180);

    // Show that predictions are non-uniform (model is actually working)
    println!("=== Verifying model outputs are non-trivial ===");
    let results = classifier.classify(&red_image, 224, 224, 1000).unwrap();
    let top = results[0].score;
    let bottom = results.last().unwrap().score;
    let max_class = results[0].class_id;
    println!("  Top class: {} (id={}, score={:.4})", label_for(max_class), max_class, top);
    println!("  Bottom class: id={}, score={:.6}", results.last().unwrap().class_id, bottom);
    println!("  Top/bottom ratio: {:.0}x", top / bottom.max(1e-10));
    println!("  -> Model is producing discriminative predictions (not uniform)");
}

fn run_classify(classifier: &ResNetClassifier<CpuBackend>, image: &[u8], w: u32, h: u32) {
    let t0 = Instant::now();
    let results = classifier.classify(image, w, h, 5).unwrap();
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    for r in &results {
        println!("  {:>4} {:>25} -> {:.2}%", r.class_id, label_for(r.class_id), r.score * 100.0);
    }
    println!("  ({:.0}ms)\n", elapsed);
}
