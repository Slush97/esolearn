//! Walk the SD 1.5 `UNet` checkpoint and report the per-stage key
//! distribution. Companion to `inspect_clip_keys.rs` for one-off
//! debugging while wiring up the loader / forward path.

use scry_diffusion::weights::SafetensorsCheckpoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = SafetensorsCheckpoint::open(
        "crates/scry-diffusion/.assets/sd-1-5/unet/diffusion_pytorch_model.safetensors",
    )?;
    let mut names = ckpt.names()?;
    names.sort();
    println!("total keys: {}", names.len());

    println!("\ntop-level (excluding down_blocks / mid_block / up_blocks / time_embedding):");
    for n in names
        .iter()
        .filter(|n| !n.starts_with("down_blocks.") && !n.starts_with("up_blocks."))
        .filter(|n| !n.starts_with("mid_block.") && !n.starts_with("time_embedding."))
    {
        println!("  {n}");
    }

    let count = |prefix: &str| names.iter().filter(|n| n.starts_with(prefix)).count();
    println!("\nper-stage counts:");
    for stage in ["down_blocks", "up_blocks"] {
        for i in 0..4 {
            let prefix = format!("{stage}.{i}.");
            let total = count(&prefix);
            let resnets = count(&format!("{prefix}resnets."));
            let attentions = count(&format!("{prefix}attentions."));
            let samplers =
                count(&format!("{prefix}downsamplers.")) + count(&format!("{prefix}upsamplers."));
            let shortcuts = names
                .iter()
                .filter(|n| n.starts_with(&prefix) && n.contains(".conv_shortcut."))
                .count();
            println!(
                "  {prefix:<22} total={total:>3}  resnets={resnets:>3}  \
                 attentions={attentions:>3}  samplers={samplers:>2}  shortcuts={shortcuts}"
            );
        }
    }
    println!("  mid_block.*            total={}", count("mid_block."));

    println!("\nleaf names under down_blocks.0.attentions.0.transformer_blocks.0.*:");
    let prefix = "down_blocks.0.attentions.0.transformer_blocks.0.";
    let mut leaves: Vec<String> = names
        .iter()
        .filter_map(|n| n.strip_prefix(prefix).map(String::from))
        .collect();
    leaves.sort();
    for l in &leaves {
        println!("  {l}");
    }

    Ok(())
}
