// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone bar chart — no feature flags needed.

use esoc_chart::v2::bar;

fn main() -> esoc_chart::error::Result<()> {
    let languages = ["Rust", "Python", "TypeScript", "Go", "Java", "C++"];
    let satisfaction = [92.0, 78.0, 73.0, 76.0, 45.0, 52.0];

    let svg = bar(&languages, &satisfaction)
        .title("Developer Satisfaction by Language")
        .to_svg()?;

    std::fs::write("basic_bar.svg", &svg)?;
    println!("Saved basic_bar.svg ({} bytes)", svg.len());
    Ok(())
}
