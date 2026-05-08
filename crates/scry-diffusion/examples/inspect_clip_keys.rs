use scry_diffusion::weights::SafetensorsCheckpoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = SafetensorsCheckpoint::open(
        "crates/scry-diffusion/.assets/sd-1-5/text_encoder/model.safetensors",
    )?;
    let mut names = ckpt.names()?;
    names.sort();
    println!("total keys: {}", names.len());
    println!("\nfirst 25:");
    for n in names.iter().take(25) {
        println!("  {n}");
    }
    println!("\n... layer 0 keys (first 12):");
    for n in names.iter().filter(|n| n.contains(".layers.0.")).take(12) {
        println!("  {n}");
    }
    println!("\n... unique layer-0 leaves:");
    let mut leaves: Vec<String> = names
        .iter()
        .filter_map(|n| n.split(".layers.0.").nth(1).map(String::from))
        .collect();
    leaves.sort();
    leaves.dedup();
    for l in &leaves {
        println!("  {l}");
    }
    println!("\n... last 10:");
    for n in names.iter().rev().take(10) {
        println!("  {n}");
    }
    Ok(())
}
