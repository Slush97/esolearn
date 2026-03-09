#![allow(
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::type_complexity,
    dead_code
)]
//! Cross-library memory footprint & cold-start benchmark.
//!
//! Compares scry-learn vs smartcore on RSS delta, fit time, and
//! first-predict latency (cold start).
//!
//! Run with:
//!   cargo run -p scry-learn --example `memory_crosslib` --release

use std::time::Instant;

fn read_rss_kb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(kb_str) = parts.get(1) {
                return kb_str.parse().unwrap_or(0);
            }
        }
    }
    0
}

struct Row {
    label: String,
    rss_kb: isize,
    fit_ms: f64,
    predict_us: f64,
}

fn gen_data(n: usize, nf: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<i32>, Vec<Vec<f64>>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = Vec::with_capacity(nf);
    for _ in 0..nf {
        let col: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
        col_major.push(col);
    }
    let target_f64: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let target_i32: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();
    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nf).map(|j| col_major[j][i]).collect())
        .collect();
    (col_major, target_f64, target_i32, row_major)
}

fn gen_reg_data(n: usize, nf: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = Vec::with_capacity(nf);
    for _ in 0..nf {
        let col: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
        col_major.push(col);
    }
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let mut y = 0.0;
            for j in 0..nf {
                y += col_major[j][i] * (j as f64 + 1.0);
            }
            y + rng.f64() * 0.5
        })
        .collect();
    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nf).map(|j| col_major[j][i]).collect())
        .collect();
    (col_major, target, row_major)
}

fn main() {
    let n = 10_000;
    let nf = 10;
    let test_n = 100;

    println!("Cross-Library Memory & Cold-Start Benchmark");
    println!("============================================");
    println!("{n} samples x {nf} features, {test_n} test rows\n");

    let mut rows: Vec<Row> = Vec::new();

    // ── Decision Tree ──
    {
        let (col_major, target_f64, target_i32, row_major) = gen_data(n, nf);
        let test = &row_major[..test_n];

        // scry
        {
            let ds = scry_learn::dataset::Dataset::new(
                col_major,
                target_f64,
                (0..nf).map(|j| format!("f{j}")).collect(),
                "y",
            );
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let mut m = scry_learn::tree::DecisionTreeClassifier::new().max_depth(10);
            m.fit(&ds).unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let t1 = Instant::now();
            let _ = std::hint::black_box(m.predict(test).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "DT  scry".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
        // smartcore
        {
            use smartcore::linalg::basic::matrix::DenseMatrix;
            let x = DenseMatrix::from_2d_vec(&row_major).unwrap();
            let params = smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                .with_max_depth(10);
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let m = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                &x,
                &target_i32,
                params,
            )
            .unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let test_x = DenseMatrix::from_2d_vec(&test.to_vec()).unwrap();
            let t1 = Instant::now();
            let _: Vec<i32> = std::hint::black_box(m.predict(&test_x).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "DT  smartcore".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
    }

    // ── Random Forest ──
    {
        let (col_major, target_f64, target_i32, row_major) = gen_data(n, nf);
        let test = &row_major[..test_n];

        // scry
        {
            let ds = scry_learn::dataset::Dataset::new(
                col_major,
                target_f64,
                (0..nf).map(|j| format!("f{j}")).collect(),
                "y",
            );
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let mut m = scry_learn::tree::RandomForestClassifier::new()
                .n_estimators(50)
                .max_depth(10)
                .seed(42);
            m.fit(&ds).unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let t1 = Instant::now();
            let _ = std::hint::black_box(m.predict(test).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "RF  scry".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
        // smartcore
        {
            use smartcore::linalg::basic::matrix::DenseMatrix;
            let x = DenseMatrix::from_2d_vec(&row_major).unwrap();
            let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(50)
                .with_max_depth(10);
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let m = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                &x,
                &target_i32,
                params,
            )
            .unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let test_x = DenseMatrix::from_2d_vec(&test.to_vec()).unwrap();
            let t1 = Instant::now();
            let _: Vec<i32> = std::hint::black_box(m.predict(&test_x).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "RF  smartcore".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
    }

    // ── Linear Regression ──
    {
        let (col_major, target, row_major) = gen_reg_data(n, nf);
        let test = &row_major[..test_n];

        // scry
        {
            let ds = scry_learn::dataset::Dataset::new(
                col_major,
                target.clone(),
                (0..nf).map(|j| format!("f{j}")).collect(),
                "y",
            );
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let mut m = scry_learn::linear::LinearRegression::new();
            m.fit(&ds).unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let t1 = Instant::now();
            let _ = std::hint::black_box(m.predict(test).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "LinReg  scry".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
        // smartcore
        {
            use smartcore::linalg::basic::matrix::DenseMatrix;
            let x = DenseMatrix::from_2d_vec(&row_major).unwrap();
            let params =
                smartcore::linear::linear_regression::LinearRegressionParameters::default();
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let m =
                smartcore::linear::linear_regression::LinearRegression::fit(&x, &target, params)
                    .unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let test_x = DenseMatrix::from_2d_vec(&test.to_vec()).unwrap();
            let t1 = Instant::now();
            let _ = std::hint::black_box(m.predict(&test_x).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "LinReg  smartcore".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
    }

    // ── KNN ──
    {
        let (col_major, target_f64, target_i32, row_major) = gen_data(n, nf);
        let test = &row_major[..test_n];

        // scry
        {
            let ds = scry_learn::dataset::Dataset::new(
                col_major,
                target_f64,
                (0..nf).map(|j| format!("f{j}")).collect(),
                "y",
            );
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let mut m = scry_learn::neighbors::KnnClassifier::new().k(5);
            m.fit(&ds).unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let t1 = Instant::now();
            let _ = std::hint::black_box(m.predict(test).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "KNN  scry".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
        // smartcore
        {
            use smartcore::linalg::basic::matrix::DenseMatrix;
            let x = DenseMatrix::from_2d_vec(&row_major).unwrap();
            let params =
                smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5);
            let rss0 = read_rss_kb();
            let t0 = Instant::now();
            let m =
                smartcore::neighbors::knn_classifier::KNNClassifier::fit(&x, &target_i32, params)
                    .unwrap();
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rss1 = read_rss_kb();
            let test_x = DenseMatrix::from_2d_vec(&test.to_vec()).unwrap();
            let t1 = Instant::now();
            let _: Vec<i32> = std::hint::black_box(m.predict(&test_x).unwrap());
            let predict_us = t1.elapsed().as_secs_f64() * 1e6;
            rows.push(Row {
                label: "KNN  smartcore".into(),
                rss_kb: rss1 as isize - rss0 as isize,
                fit_ms,
                predict_us,
            });
        }
    }

    // ── Print results ──
    println!(
        "{:<20} {:>10} {:>12} {:>14}",
        "Model", "RSS delta", "Fit (ms)", "Cold predict"
    );
    println!("{:-<58}", "");

    let labels = ["DT", "RF", "LinReg", "KNN"];
    for (idx, pair) in rows.chunks(2).enumerate() {
        let scry = &pair[0];
        let other = &pair[1];

        let fmt_rss = |kb: isize| -> String {
            if kb.abs() < 1024 {
                format!("{kb} KB")
            } else {
                format!("{:.1} MB", kb as f64 / 1024.0)
            }
        };
        let fmt_pred = |us: f64| -> String {
            if us < 1000.0 {
                format!("{us:.0}us")
            } else {
                format!("{:.1}ms", us / 1000.0)
            }
        };

        println!(
            "{:<20} {:>10} {:>12.2} {:>14}",
            format!("{} scry", labels[idx]),
            fmt_rss(scry.rss_kb),
            scry.fit_ms,
            fmt_pred(scry.predict_us)
        );
        println!(
            "{:<20} {:>10} {:>12.2} {:>14}",
            format!("{} smartcore", labels[idx]),
            fmt_rss(other.rss_kb),
            other.fit_ms,
            fmt_pred(other.predict_us)
        );
        // speedup line
        let fit_ratio = other.fit_ms / scry.fit_ms.max(0.001);
        let pred_ratio = other.predict_us / scry.predict_us.max(0.001);
        println!(
            "{:<20} {:>10} {:>12} {:>14}",
            "",
            "---",
            format!("{fit_ratio:.1}x"),
            format!("{pred_ratio:.1}x"),
        );
        println!();
    }
}
