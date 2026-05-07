//! Rust port of the Jake VanderPlas PDSH 05_02 final-project add-ons.
//!
//! Three steps:
//!   1. PCA on the iris dataset — print the singular values.
//!   2. PCA on the digits dataset — render a screegram of singular values.
//!   3. KNN sweep over `k` ∈ 1..=10 on a held-out digits split — render
//!      the validation-accuracy curve and report the best `k`.

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use esoc_chart::new_theme::NewTheme;
use esoc_chart::v2;

use scry_learn::dataset::Dataset;
use scry_learn::metrics::accuracy;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{Pca, Transformer};
use scry_learn::split::train_test_split;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("examples/<crate> lives two levels under workspace root")
        .to_path_buf()
}

fn fixtures_dir() -> PathBuf {
    workspace_root().join("crates/scry-learn/tests/fixtures")
}

fn figures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("report")
        .join("figures")
}

/// Read a CSV (header + numeric body) as column-major f64 columns.
fn read_csv_f64(path: &PathBuf) -> Result<(Vec<String>, Vec<Vec<f64>>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| format!("empty csv: {}", path.display()))?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); header.len()];
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        for (i, cell) in line.split(',').enumerate() {
            cols[i].push(cell.trim().parse::<f64>()?);
        }
    }
    Ok((header, cols))
}

fn load_iris() -> Result<Dataset, Box<dyn Error>> {
    let (names, cols) = read_csv_f64(&fixtures_dir().join("iris_features.csv"))?;
    let n = cols[0].len();
    Ok(Dataset::new(cols, vec![0.0; n], names, "label"))
}

fn load_digits() -> Result<Dataset, Box<dyn Error>> {
    let dir = fixtures_dir();
    let (feat_names, feat_cols) = read_csv_f64(&dir.join("digits_features.csv"))?;
    let (_, mut tgt_cols) = read_csv_f64(&dir.join("digits_target.csv"))?;
    let target = tgt_cols
        .pop()
        .ok_or("digits_target.csv has no columns")?;
    Ok(Dataset::new(feat_cols, target, feat_names, "digit"))
}

/// scry-learn's PCA exposes eigenvalues of the sample covariance matrix.
/// sklearn's `singular_values_` are the singular values of the centred data
/// matrix. The relationship is σᵢ = √(λᵢ · (n − 1)).
fn singular_values(pca: &Pca, n_samples: usize) -> Vec<f64> {
    let denom = (n_samples - 1) as f64;
    pca.explained_variance()
        .iter()
        .map(|v| (v * denom).sqrt())
        .collect()
}

/// Convert a Dataset's column-major features to row-major rows for predict().
fn rows(data: &Dataset) -> Vec<Vec<f64>> {
    (0..data.n_samples()).map(|i| data.sample(i)).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let figs = figures_dir();
    fs::create_dir_all(&figs)?;
    let theme = NewTheme::light();

    // ── 1. PCA on iris ─────────────────────────────────────────────
    let iris = load_iris()?;
    let mut pca = Pca::new();
    pca.fit(&iris)?;
    let iris_sv = singular_values(&pca, iris.n_samples());

    println!("=== iris PCA ===");
    println!("  n_samples={}, n_features={}", iris.n_samples(), iris.n_features());
    for (i, s) in iris_sv.iter().enumerate() {
        println!("  sv[{i}] = {s:.6}");
    }
    println!("  largest singular value     = {:.6}", iris_sv[0]);
    println!("  2nd largest singular value = {:.6}", iris_sv[1]);

    // ── 2. PCA on digits — screegram ───────────────────────────────
    let digits = load_digits()?;
    let mut dpca = Pca::new();
    dpca.fit(&digits)?;
    let dsv = singular_values(&dpca, digits.n_samples());
    let dsv_x: Vec<f64> = (0..dsv.len()).map(|i| i as f64).collect();

    let screegram_path = figs.join("digits_singular_values.svg");
    v2::line(&dsv_x, &dsv)
        .title("Singular values — digits dataset")
        .x_label("component index")
        .y_label("singular value")
        .theme(theme.clone())
        .save_svg(screegram_path.to_str().unwrap())?;

    println!("\n=== digits PCA ===");
    println!("  n_samples={}, n_features={}", digits.n_samples(), digits.n_features());
    println!("  largest singular value = {:.4}", dsv[0]);
    println!("  wrote {}", screegram_path.display());

    // ── 3. KNN sweep on digits ────────────────────────────────────
    let (train, test) = train_test_split(&digits, 0.25, 42);
    let test_x = rows(&test);
    let test_y = test.target.clone();

    println!(
        "\n=== KNN sweep (train={}, test={}) ===",
        train.n_samples(),
        test.n_samples()
    );

    let ks: Vec<f64> = (1..=10).map(|k| k as f64).collect();
    let mut accs: Vec<f64> = Vec::with_capacity(10);
    for k in 1..=10usize {
        let mut knn = KnnClassifier::new().k(k);
        knn.fit(&train)?;
        let yhat = knn.predict(&test_x)?;
        let a = accuracy(&test_y, &yhat);
        accs.push(a);
        println!("  k={k:2}  accuracy={a:.4}");
    }

    let (best_idx, best_acc) = accs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let best_k = best_idx + 1;
    println!("\n  best k = {best_k}   accuracy = {best_acc:.4}");

    let knn_path = figs.join("knn_validation_accuracy.svg");
    v2::line(&ks, &accs)
        .title(&format!("KNN validation accuracy — digits (best k = {best_k})"))
        .x_label("n_neighbors")
        .y_label("accuracy")
        .theme(theme)
        .save_svg(knn_path.to_str().unwrap())?;
    println!("  wrote {}", knn_path.display());

    Ok(())
}
