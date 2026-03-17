// SPDX-License-Identifier: MIT OR Apache-2.0
//! Basic treemap example: sales by product category.

fn main() -> esoc_chart::error::Result<()> {
    let labels = vec![
        "Electronics",
        "Clothing",
        "Food & Beverage",
        "Home & Garden",
        "Books",
        "Sports",
        "Toys",
        "Automotive",
    ];
    let values = vec![320.0, 180.0, 150.0, 120.0, 80.0, 60.0, 45.0, 30.0];

    let mut theme = esoc_chart::new_theme::NewTheme::light();
    theme.base_font_size = 13.0;
    theme.title_font_size = 18.0;
    theme.legend_font_size = 11.0;
    theme.font_family = "Inter, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif".into();

    let svg = esoc_chart::express::treemap(&labels, &values)
        .title("Sales by Category ($M)")
        .theme(theme)
        .size(900.0, 600.0)
        .to_svg()?;

    std::fs::write("treemap.svg", &svg)?;
    println!("Wrote treemap.svg ({} bytes)", svg.len());
    Ok(())
}
