// SPDX-License-Identifier: MIT OR Apache-2.0
//! Treemap of esolearn workspace crates by source LOC.

use esoc_chart::{express::treemap, new_theme::NewTheme};

fn main() -> esoc_chart::error::Result<()> {
    let crates = [
        ("scry-learn",     44_987.0),
        ("scry-llm",       12_743.0),
        ("scry-vision",    12_201.0),
        ("esoc-chart",      9_559.0),
        ("scry-cv",         9_240.0),
        ("scry-gpu",        6_146.0),
        ("scry-diffusion",  5_236.0),
        ("scry-stt",        3_772.0),
        ("esoc-scene",      2_247.0),
        ("esoc-geo",        1_857.0),
        ("esoc-color",      1_065.0),
        ("esoc-gfx",          709.0),
        ("scry-app",          662.0),
    ];

    let labels: Vec<&str> = crates.iter().map(|(n, _)| *n).collect();
    let values: Vec<f64>  = crates.iter().map(|(_, v)| *v).collect();

    let mut theme = NewTheme::light();
    theme.base_font_size = 12.0;
    theme.title_font_size = 18.0;
    theme.font_family = "Inter, -apple-system, \"Segoe UI\", Helvetica, sans-serif".into();

    let svg = treemap(&labels, &values)
        .title("esolearn workspace — source LOC by crate")
        .theme(theme)
        .size(1100.0, 700.0)
        .to_svg()?;

    let out = std::env::args().nth(1).unwrap_or_else(|| "codebase_treemap.svg".into());
    std::fs::write(&out, &svg)?;
    println!("Wrote {} ({} bytes)", out, svg.len());
    Ok(())
}
