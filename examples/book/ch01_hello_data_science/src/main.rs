use esoc_chart::express::scatter;
use scry_learn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dataset
    let ds = Dataset::from_csv("datasets/iris/iris.csv", "species")?;

    // Print summary statistics
    println!("Iris Dataset Summary");
    println!("====================\n");
    ds.describe();

    // Map encoded target back to species names
    let labels = ds.class_labels.as_ref().expect("iris has class labels");
    let species: Vec<String> = ds.target.iter().map(|&t| labels[t as usize].clone()).collect();

    // Scatter plot: sepal length vs sepal width, colored by species
    let svg = scatter(ds.feature(0), ds.feature(1))
        .color_by(&species)
        .title("Iris: Sepal Length vs Sepal Width")
        .x_label("Sepal Length (cm)")
        .y_label("Sepal Width (cm)")
        .to_svg()?;

    std::fs::write("iris_scatter.svg", &svg)?;
    println!("\nSaved scatter plot to iris_scatter.svg");

    Ok(())
}
