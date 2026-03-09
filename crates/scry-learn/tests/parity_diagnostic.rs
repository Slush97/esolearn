//! Quick diagnostic: compare scry-learn predictions vs sklearn golden references
//! on full datasets to determine if accuracy gaps are model bugs or eval bugs.
//!
//! Run: cargo run --example `parity_diagnostic` -p scry-learn

fn main() {
    let fixtures = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");

    // Load sklearn predictions
    let json_str = std::fs::read_to_string(fixtures.join("sklearn_predictions.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    println!("═══════════════════════════════════════════════════════");
    println!("  PARITY DIAGNOSTIC: scry-learn vs sklearn predictions");
    println!("═══════════════════════════════════════════════════════\n");

    // ── Iris DecisionTree ──
    {
        let (features, target) = load_csv(&fixtures, "iris");
        let feat_names: Vec<String> = (0..features.len()).map(|i| format!("f{i}")).collect();
        let ds = scry_learn::dataset::Dataset::new(
            features.clone(),
            target.clone(),
            feat_names,
            "target",
        );

        let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(5);
        dt.fit(&ds).unwrap();

        // Predict on all samples (row-major)
        let test_rows = to_rows(&features);
        let preds = dt.predict(&test_rows).unwrap();

        let sklearn_preds: Vec<f64> = json["dt_iris"]["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let sklearn_acc: f64 = json["dt_iris"]["accuracy"].as_f64().unwrap();

        compare(
            "Iris DT (max_depth=5)",
            &preds,
            &sklearn_preds,
            &target,
            sklearn_acc,
        );
    }

    // ── Iris KNN ──
    {
        let (features, target) = load_csv(&fixtures, "iris");
        let feat_names: Vec<String> = (0..features.len()).map(|i| format!("f{i}")).collect();
        let ds = scry_learn::dataset::Dataset::new(
            features.clone(),
            target.clone(),
            feat_names,
            "target",
        );

        let mut knn = scry_learn::neighbors::KnnClassifier::new().k(5);
        knn.fit(&ds).unwrap();

        let test_rows = to_rows(&features);
        let preds = knn.predict(&test_rows).unwrap();

        let sklearn_preds: Vec<f64> = json["knn_iris"]["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let sklearn_acc: f64 = json["knn_iris"]["accuracy"].as_f64().unwrap();

        compare(
            "Iris KNN (k=5)",
            &preds,
            &sklearn_preds,
            &target,
            sklearn_acc,
        );
    }

    // ── Wine DecisionTree ──
    {
        let (features, target) = load_csv(&fixtures, "wine");
        let feat_names: Vec<String> = (0..features.len()).map(|i| format!("f{i}")).collect();
        let ds = scry_learn::dataset::Dataset::new(
            features.clone(),
            target.clone(),
            feat_names,
            "target",
        );

        let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(5);
        dt.fit(&ds).unwrap();

        let test_rows = to_rows(&features);
        let preds = dt.predict(&test_rows).unwrap();

        let sklearn_preds: Vec<f64> = json["dt_wine"]["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let sklearn_acc: f64 = json["dt_wine"]["accuracy"].as_f64().unwrap();

        compare(
            "Wine DT (max_depth=5)",
            &preds,
            &sklearn_preds,
            &target,
            sklearn_acc,
        );
    }

    // ── Iris LogReg (with scaling) ──
    {
        let (features, target) = load_csv(&fixtures, "iris");
        let feat_names: Vec<String> = (0..features.len()).map(|i| format!("f{i}")).collect();

        // Scale features first (sklearn used StandardScaler)
        let mut ds = scry_learn::dataset::Dataset::new(
            features,
            target.clone(),
            feat_names,
            "target",
        );
        let mut scaler = scry_learn::preprocess::StandardScaler::new();
        scry_learn::preprocess::Transformer::fit(&mut scaler, &ds).unwrap();
        scry_learn::preprocess::Transformer::transform(&scaler, &mut ds).unwrap();

        let mut lr = scry_learn::linear::LogisticRegression::new().max_iter(200);
        lr.fit(&ds).unwrap();

        let test_rows = to_rows(&ds.features);
        let preds = lr.predict(&test_rows).unwrap();

        let sklearn_preds: Vec<f64> = json["logreg_iris"]["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let sklearn_acc: f64 = json["logreg_iris"]["accuracy"].as_f64().unwrap();

        compare(
            "Iris LogReg (max_iter=200, scaled)",
            &preds,
            &sklearn_preds,
            &target,
            sklearn_acc,
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════");
    println!("If train-set predictions match sklearn closely (>98%),");
    println!("the accuracy gap is purely k-fold evaluation methodology.");
    println!("If predictions diverge, model implementations need auditing.");
}

fn load_csv(fixtures: &std::path::Path, name: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let feat_path = fixtures.join(format!("{name}_features.csv"));
    let target_path = fixtures.join(format!("{name}_target.csv"));

    // Load features (CSV → column-major)
    let mut rdr = csv::Reader::from_path(&feat_path).unwrap();
    let n_cols = rdr.headers().unwrap().len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        rows.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect());
    }
    let n_rows = rows.len();
    let mut cols = vec![vec![0.0; n_rows]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }

    // Load target
    let mut rdr = csv::Reader::from_path(&target_path).unwrap();
    let target: Vec<f64> = rdr
        .records()
        .map(|r| r.unwrap()[0].parse::<f64>().unwrap())
        .collect();

    (cols, target)
}

fn to_rows(col_major: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = col_major[0].len();
    (0..n)
        .map(|i| col_major.iter().map(|col| col[i]).collect())
        .collect()
}

fn compare(
    label: &str,
    scry_preds: &[f64],
    sklearn_preds: &[f64],
    target: &[f64],
    sklearn_train_acc: f64,
) {
    let n = target.len();
    assert_eq!(scry_preds.len(), n);
    assert_eq!(sklearn_preds.len(), n);

    let scry_correct = scry_preds
        .iter()
        .zip(target)
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let sklearn_correct = sklearn_preds
        .iter()
        .zip(target)
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let pred_match = scry_preds
        .iter()
        .zip(sklearn_preds)
        .filter(|(s, sk)| (**s - **sk).abs() < 0.5)
        .count();

    // Find divergent samples
    let divergent: Vec<usize> = (0..n)
        .filter(|&i| (scry_preds[i] - sklearn_preds[i]).abs() >= 0.5)
        .collect();

    println!("── {label} ──");
    println!("  Samples:           {n}");
    println!(
        "  scry-learn acc:    {:.1}% ({}/{})",
        scry_correct as f64 / n as f64 * 100.0,
        scry_correct,
        n
    );
    println!(
        "  sklearn acc:       {:.1}% ({}/{}) [stored: {:.1}%]",
        sklearn_correct as f64 / n as f64 * 100.0,
        sklearn_correct,
        n,
        sklearn_train_acc * 100.0
    );
    println!(
        "  Prediction match:  {:.1}% ({}/{})",
        pred_match as f64 / n as f64 * 100.0,
        pred_match,
        n
    );

    if divergent.is_empty() {
        println!("  ✅ IDENTICAL predictions — gap is purely eval methodology");
    } else {
        println!(
            "  ⚠️  {} DIVERGENT samples: {:?}",
            divergent.len(),
            if divergent.len() <= 20 {
                &divergent[..]
            } else {
                &divergent[..20]
            }
        );
        // Show first few divergences
        for &i in divergent.iter().take(5) {
            println!(
                "     sample {i}: scry={:.0} sklearn={:.0} true={:.0}",
                scry_preds[i], sklearn_preds[i], target[i]
            );
        }
    }
    println!();
}
