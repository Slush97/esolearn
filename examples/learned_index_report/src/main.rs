// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generate all charts for the learned index report.

use esoc_chart::v2::*;

fn main() -> esoc_chart::error::Result<()> {
    let out = "examples/learned_index_report/figures";
    std::fs::create_dir_all(out).unwrap();
    let theme = NewTheme::publication();

    // ── Figure 1: CDF of uniform vs clustered keys ──────────────────
    // Uniform keys: positions are a straight line
    let n = 200;
    let uniform_keys: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let uniform_pos: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    let svg = line(&uniform_keys, &uniform_pos)
        .title("CDF of Uniformly Distributed Keys")
        .x_label("Key")
        .y_label("Position (normalized)")
        .theme(theme.clone())
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/cdf_uniform.svg"), &svg)?;
    println!("Saved cdf_uniform.svg");

    // Clustered keys: S-curve CDF (logistic-shaped)
    let clustered_keys: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        // Mix of two clusters centered at 0.3 and 0.7
        let c1 = (-20.0 * (t - 0.3)).exp();
        let c2 = (-20.0 * (t - 0.7)).exp();
        t + 0.15 / (1.0 + c1) + 0.15 / (1.0 + c2)
    }).collect();
    // Normalize to [0, 1]
    let ck_min = clustered_keys.iter().cloned().fold(f64::INFINITY, f64::min);
    let ck_max = clustered_keys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let clustered_pos: Vec<f64> = clustered_keys.iter()
        .map(|&k| (k - ck_min) / (ck_max - ck_min))
        .collect();
    let clustered_x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let svg = line(&clustered_x, &clustered_pos)
        .title("CDF of Clustered Keys")
        .x_label("Key")
        .y_label("Position (normalized)")
        .theme(theme.clone())
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/cdf_clustered.svg"), &svg)?;
    println!("Saved cdf_clustered.svg");

    // ── Figure 2: Linear model fit vs actual CDF ────────────────────
    // Show a linear approximation overlaid on the clustered CDF
    // Using two layers via Grammar API
    let linear_approx: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

    let chart = Chart::new()
        .layer(
            Layer::new(MarkType::Line)
                .with_x(clustered_x.clone())
                .with_y(clustered_pos.clone())
                .with_label("Actual CDF"),
        )
        .layer(
            Layer::new(MarkType::Line)
                .with_x(clustered_x.clone())
                .with_y(linear_approx)
                .with_label("Linear Model"),
        )
        .title("Single Linear Model vs Actual CDF")
        .x_label("Key")
        .y_label("Position (normalized)")
        .theme(theme.clone())
        .size(700.0, 400.0);
    let svg = chart.to_svg()?;
    std::fs::write(format!("{out}/linear_vs_cdf.svg"), &svg)?;
    println!("Saved linear_vs_cdf.svg");

    // ── Figure 3: Prediction error of single linear model ───────────
    let errors: Vec<f64> = clustered_pos.iter().enumerate()
        .map(|(i, &actual)| {
            let predicted = i as f64 / (n - 1) as f64;
            (actual - predicted) * n as f64 // error in number of positions
        })
        .collect();
    let error_x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let chart = Chart::new()
        .layer(
            Layer::new(MarkType::Area)
                .with_x(error_x.clone())
                .with_y(errors.clone()),
        )
        .annotate(Annotation::hline(0.0))
        .title("Prediction Error of Single Linear Model")
        .x_label("Key")
        .y_label("Error (positions)")
        .theme(theme.clone())
        .size(700.0, 400.0);
    let svg = chart.to_svg()?;
    std::fs::write(format!("{out}/linear_error.svg"), &svg)?;
    println!("Saved linear_error.svg");

    // ── Figure 4: Piecewise linear approximation (PGM concept) ──────
    // Break into segments, fit linear model per segment
    let num_segments = 8;
    let seg_size = n / num_segments;
    let mut piecewise_approx: Vec<f64> = Vec::with_capacity(n);
    for seg in 0..num_segments {
        let start = seg * seg_size;
        let end = if seg == num_segments - 1 { n } else { (seg + 1) * seg_size };
        let y_start = clustered_pos[start];
        let y_end = clustered_pos[end - 1];
        let count = end - start;
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            piecewise_approx.push(y_start + t * (y_end - y_start));
        }
    }

    let chart = Chart::new()
        .layer(
            Layer::new(MarkType::Line)
                .with_x(clustered_x.clone())
                .with_y(clustered_pos.clone())
                .with_label("Actual CDF"),
        )
        .layer(
            Layer::new(MarkType::Line)
                .with_x(clustered_x.clone())
                .with_y(piecewise_approx.clone())
                .with_label("Piecewise Linear (8 segments)"),
        )
        .title("Piecewise Linear Approximation (PGM-Index)")
        .x_label("Key")
        .y_label("Position (normalized)")
        .theme(theme.clone())
        .size(700.0, 400.0);
    let svg = chart.to_svg()?;
    std::fs::write(format!("{out}/pgm_fit.svg"), &svg)?;
    println!("Saved pgm_fit.svg");

    // ── Figure 5: Piecewise error vs linear error ───────────────────
    let pw_errors: Vec<f64> = clustered_pos.iter().enumerate()
        .map(|(i, &actual)| {
            let predicted = piecewise_approx[i];
            (actual - predicted).abs() * n as f64
        })
        .collect();
    let lin_errors_abs: Vec<f64> = errors.iter().map(|e| e.abs()).collect();

    let chart = Chart::new()
        .layer(
            Layer::new(MarkType::Line)
                .with_x(error_x.clone())
                .with_y(lin_errors_abs)
                .with_label("Single Linear Model"),
        )
        .layer(
            Layer::new(MarkType::Line)
                .with_x(error_x.clone())
                .with_y(pw_errors)
                .with_label("PGM (8 segments)"),
        )
        .title("Absolute Prediction Error Comparison")
        .x_label("Key")
        .y_label("|Error| (positions)")
        .theme(theme.clone())
        .size(700.0, 400.0);
    let svg = chart.to_svg()?;
    std::fs::write(format!("{out}/error_comparison.svg"), &svg)?;
    println!("Saved error_comparison.svg");

    // ── Figure 6: Lookup cost comparison (bar chart) ────────────────
    let methods = ["B-Tree", "Hash Map", "Binary Search", "Learned (PGM)"];
    // Relative costs: comparisons or memory accesses per lookup (illustrative)
    let comparisons = [20.0, 1.0, 20.0, 3.0];
    let svg = bar(&methods, &comparisons)
        .title("Lookup Cost: Comparisons per Query (1M keys)")
        .y_label("Avg. Comparisons / Memory Accesses")
        .theme(theme.clone())
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/lookup_cost.svg"), &svg)?;
    println!("Saved lookup_cost.svg");

    // ── Figure 7: Memory usage comparison (bar chart) ───────────────
    let mem_methods = ["B-Tree", "Hash Map", "Sorted Array", "PGM-Index"];
    // Approximate bytes per key for 1M 64-bit integer keys
    let mem_bytes = [56.0, 72.0, 8.0, 8.2];
    let svg = bar(&mem_methods, &mem_bytes)
        .title("Memory Overhead per Key (64-bit integers)")
        .y_label("Bytes per Key")
        .theme(theme.clone())
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/memory_usage.svg"), &svg)?;
    println!("Saved memory_usage.svg");

    // ── Figure 8: Cache misses illustration ─────────────────────────
    // Show how B-tree traversal depth relates to cache misses
    let dataset_sizes: Vec<f64> = (10..=28).map(|p| 2.0_f64.powi(p)).collect();
    let btree_misses: Vec<f64> = dataset_sizes.iter()
        .map(|&n| (n.log2() / 4.0).ceil()) // ~log_B(n) with fanout B≈16
        .collect();
    let learned_misses: Vec<f64> = dataset_sizes.iter()
        .map(|&_n| 2.0) // model eval (L1 cache) + 1 correction scan
        .collect();
    let dataset_sizes_log: Vec<f64> = dataset_sizes.iter().map(|n| n.log2()).collect();

    let chart = Chart::new()
        .layer(
            Layer::new(MarkType::Line)
                .with_x(dataset_sizes_log.clone())
                .with_y(btree_misses)
                .with_label("B-Tree"),
        )
        .layer(
            Layer::new(MarkType::Line)
                .with_x(dataset_sizes_log)
                .with_y(learned_misses)
                .with_label("Learned Index"),
        )
        .title("Expected Cache Misses vs Dataset Size")
        .x_label("Dataset Size (log₂ N)")
        .y_label("Cache Misses per Lookup")
        .theme(theme.clone())
        .size(700.0, 400.0);
    let svg = chart.to_svg()?;
    std::fs::write(format!("{out}/cache_misses.svg"), &svg)?;
    println!("Saved cache_misses.svg");

    // ── Figure 9: Error bound ε vs number of segments ───────────────
    // Simulated tradeoff curve
    let epsilons: Vec<f64> = vec![2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
    // More segments needed for tighter bounds (inverse relationship)
    let segments: Vec<f64> = epsilons.iter()
        .map(|&eps| (1000.0 / eps).ceil())
        .collect();

    let svg = line(&epsilons, &segments)
        .title("Error Bound ε vs Number of Segments")
        .x_label("ε (max error in positions)")
        .y_label("Number of Segments")
        .theme(theme.clone())
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/epsilon_tradeoff.svg"), &svg)?;
    println!("Saved epsilon_tradeoff.svg");

    // ── Figure 10: Histogram of key distributions ───────────────────
    // Generate clustered data to show non-uniform distribution
    let mut clustered_data: Vec<f64> = Vec::new();
    let mut seed: u64 = 42;
    for _ in 0..500 {
        // Simple LCG PRNG
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (seed >> 33) as f64 / (1u64 << 31) as f64;
        // Bimodal: cluster around 25 and 75
        if u < 0.5 {
            clustered_data.push(25.0 + (u - 0.25) * 40.0);
        } else {
            clustered_data.push(75.0 + (u - 0.75) * 40.0);
        }
    }

    let svg = histogram(&clustered_data)
        .bins(20)
        .title("Bimodal Key Distribution")
        .x_label("Key Value")
        .y_label("Frequency")
        .theme(theme)
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write(format!("{out}/key_distribution.svg"), &svg)?;
    println!("Saved key_distribution.svg");

    println!("\nAll figures generated in {out}/");
    Ok(())
}
