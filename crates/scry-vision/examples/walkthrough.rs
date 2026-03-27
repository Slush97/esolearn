// SPDX-License-Identifier: MIT OR Apache-2.0
//! scry-vision walkthrough — what this library does and how the pieces fit.
//!
//! Run with: cargo run -p scry-vision --example walkthrough
//!
//! This example uses zero-initialized weights, so model outputs are
//! meaningless — the point is to show the data flow, not real predictions.

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::Tensor;

use scry_vision::image::ImageBuffer;
use scry_vision::models::{ResNet, ResNetClassifier, ResNetConfig};
use scry_vision::pipeline::Classify;
use scry_vision::postprocess::classify::top_k_softmax;
use scry_vision::postprocess::embedding::cosine_similarity;
use scry_vision::transform::resize::{InterpolationMode, Resize};
use scry_vision::transform::to_tensor::ToTensor;
use scry_vision::transform::ImageTransform;

fn main() {
    println!("=== scry-vision walkthrough ===\n");
    println!("scry-vision is an inference toolkit for computer vision models.");
    println!("It handles the full pipeline: image in -> structured predictions out.\n");

    // -----------------------------------------------------------------------
    // 1. IMAGES
    // -----------------------------------------------------------------------
    println!("--- 1. Images ---\n");
    println!("Everything starts with raw pixel data. ImageBuffer holds HWC u8 pixels.");
    println!("In a real app you'd decode a JPEG/PNG; here we synthesize two images.\n");

    // Simulate a 64x64 "reddish" photo and a "bluish" photo.
    let mut red_pixels = vec![0u8; 64 * 64 * 3];
    let mut blue_pixels = vec![0u8; 64 * 64 * 3];
    for i in 0..(64 * 64) {
        // Red image: warm tones
        red_pixels[i * 3] = 200;     // R
        red_pixels[i * 3 + 1] = 80;  // G
        red_pixels[i * 3 + 2] = 60;  // B
        // Blue image: cool tones
        blue_pixels[i * 3] = 40;     // R
        blue_pixels[i * 3 + 1] = 90; // G
        blue_pixels[i * 3 + 2] = 210; // B
    }

    let red_img = ImageBuffer::from_raw(red_pixels.clone(), 64, 64, 3).unwrap();
    let blue_img = ImageBuffer::from_raw(blue_pixels.clone(), 64, 64, 3).unwrap();

    println!("  red_img:  {}x{}, {} channels, {} bytes",
        red_img.width, red_img.height, red_img.channels, red_img.byte_len());
    println!("  blue_img: {}x{}, {} channels, {} bytes\n",
        blue_img.width, blue_img.height, blue_img.channels, blue_img.byte_len());

    // -----------------------------------------------------------------------
    // 2. TRANSFORMS
    // -----------------------------------------------------------------------
    println!("--- 2. Transforms (preprocessing) ---\n");
    println!("Models expect a specific input format. Transforms convert raw images");
    println!("into what the model needs. Common steps:\n");
    println!("  Resize      -> fixed dimensions (e.g., 224x224)");
    println!("  ToTensor     -> HWC u8 [0,255] to CHW f32 [0,1]");
    println!("  Normalize    -> subtract mean, divide by std (per channel)\n");

    let resize = Resize::new(224, 224, InterpolationMode::Bilinear);
    let resized = resize.apply(&red_img).unwrap();
    println!("  After resize: {}x{}", resized.width, resized.height);

    let tensor: Tensor<CpuBackend> = ToTensor::imagenet().apply(&resized);
    println!("  After ToTensor::imagenet(): shape {:?}, dtype f32", tensor.shape.dims());
    println!("    (ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])");

    let sample: Vec<f32> = tensor.to_vec().into_iter().take(5).collect();
    println!("    First 5 values: {:?}\n", sample);

    // -----------------------------------------------------------------------
    // 3. CLASSIFICATION — "what is in this image?"
    // -----------------------------------------------------------------------
    println!("--- 3. Classification pipeline ---\n");
    println!("Classification answers: \"what is in this image?\"");
    println!("The model outputs a score for each possible class (e.g., 1000 ImageNet classes).");
    println!("Pipeline: Resize -> Normalize -> ResNet forward -> softmax -> top-k\n");

    let config = ResNetConfig::resnet18(1000);
    let classifier = ResNetClassifier::<CpuBackend>::new(ResNet::new(config));

    let results = classifier.classify(&red_img.data, 64, 64, 5).unwrap();

    println!("  Top-5 predictions for red_img (zero weights — scores will be uniform):");
    for r in &results {
        println!("    class {:>4} -> {:.4}", r.class_id, r.score);
    }
    println!();

    // Show what it looks like with non-uniform logits (simulating a real model)
    println!("  What it looks like with a real model's logits:");
    let mut fake_logits = vec![0.0f32; 5];
    fake_logits[0] = 8.2;  // "golden retriever"
    fake_logits[1] = 5.1;  // "labrador"
    fake_logits[2] = 1.0;  // "tennis ball"
    fake_logits[3] = 0.3;  // "grass"
    fake_logits[4] = -2.0; // "car"
    let labels = ["golden retriever", "labrador", "tennis ball", "grass", "car"];
    let preds = top_k_softmax(&fake_logits, 5);
    for p in &preds {
        println!("    {:>18} -> {:.1}%", labels[p.class_id as usize], p.score * 100.0);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. DETECTION — "where are objects in this image?"
    // -----------------------------------------------------------------------
    println!("--- 4. Detection pipeline ---\n");
    println!("Detection answers: \"where are the objects and what are they?\"");
    println!("Output is a list of bounding boxes, each with a class and confidence.\n");
    println!("  Example output (from a real YOLO model):");
    println!("    Detection {{ bbox: [120, 45, 380, 290], class: 'person', conf: 0.94 }}");
    println!("    Detection {{ bbox: [400, 200, 520, 350], class: 'dog',    conf: 0.87 }}");
    println!();
    println!("  scry-vision provides YoloDetector (v8/v11) and ScrfdDetector (faces).");
    println!("  Pipeline: Letterbox -> ToTensor -> model -> decode boxes -> NMS -> rescale\n");
    println!("  NMS (non-maximum suppression) removes duplicate overlapping detections —");
    println!("  if 5 boxes all see the same dog, NMS keeps the best one.\n");

    // -----------------------------------------------------------------------
    // 5. EMBEDDINGS — "how similar are these images?"
    // -----------------------------------------------------------------------
    println!("--- 5. Embedding pipeline ---\n");
    println!("Embeddings answer: \"how similar are two images?\"");
    println!("The model compresses an image into a fixed-size vector (e.g., 512 floats).");
    println!("Similar images produce vectors that point in the same direction.\n");

    // Demonstrate with synthetic embeddings (since zero weights won't differ)
    let embed_a = vec![0.8, 0.5, 0.2, 0.1];
    let embed_b = vec![0.75, 0.55, 0.18, 0.12]; // similar to A
    let embed_c = vec![-0.3, 0.1, 0.9, -0.5];   // very different

    println!("  Embedding A (photo of a cat):  {:?}", embed_a);
    println!("  Embedding B (photo of a cat):  {:?}", embed_b);
    println!("  Embedding C (photo of a truck): {:?}\n", embed_c);

    let sim_ab = cosine_similarity(&embed_a, &embed_b);
    let sim_ac = cosine_similarity(&embed_a, &embed_c);
    println!("  cosine_similarity(A, B) = {:.4}  <- similar images, high score", sim_ab);
    println!("  cosine_similarity(A, C) = {:.4}  <- different images, low score\n", sim_ac);

    println!("  Use cases: face recognition, image search, duplicate detection.");
    println!("  scry-vision provides ClipEmbedder (general) and ArcFaceEmbedder (faces).\n");

    // -----------------------------------------------------------------------
    // 6. PUTTING IT TOGETHER
    // -----------------------------------------------------------------------
    println!("--- 6. Architecture summary ---\n");
    println!("  ImageBuffer          raw pixels (the input)");
    println!("       |");
    println!("  transform/           preprocessing (resize, normalize, pad, crop)");
    println!("       |");
    println!("  model/               inference backend (native tensors or ONNX)");
    println!("       |");
    println!("  postprocess/         decode raw output (NMS, softmax, L2 norm)");
    println!("       |");
    println!("  pipeline traits      user-facing API (Detect, Classify, Embed)");
    println!("       |");
    println!("  models/              pre-built pipelines that wire it all together");
    println!("                       (YoloDetector, ResNetClassifier, ClipEmbedder, ...)\n");

    println!("With real weights loaded (via safetensors or ONNX), these produce");
    println!("actual predictions. The zero-weight models above just show the plumbing.");
}
