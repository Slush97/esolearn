// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for model schema versioning.

#[cfg(feature = "serde")]
mod serde_tests {
    use scry_learn::dataset::Dataset;
    use scry_learn::error::ScryLearnError;
    use scry_learn::linear::LinearRegression;

    fn make_dataset() -> Dataset {
        Dataset::new(
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec!["x".into()],
            "y",
        )
    }

    #[test]
    fn roundtrip_serialization_succeeds() {
        let data = make_dataset();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();

        let json = serde_json::to_string(&model).unwrap();
        let loaded: LinearRegression = serde_json::from_str(&json).unwrap();

        let preds = loaded.predict(&[vec![3.0]]).unwrap();
        assert!(
            preds[0] > 5.0 && preds[0] < 7.0,
            "prediction should be ~6.0, got {}",
            preds[0]
        );
    }

    #[test]
    fn wrong_schema_version_rejected() {
        let data = make_dataset();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();

        let json = serde_json::to_string(&model).unwrap();

        // Tamper with the schema version.
        let tampered = json.replace("\"_schema_version\":1", "\"_schema_version\":999");
        assert_ne!(json, tampered, "should have replaced version");

        let loaded: LinearRegression = serde_json::from_str(&tampered).unwrap();
        let err = loaded.predict(&[vec![3.0]]).unwrap_err();
        assert!(
            matches!(err, ScryLearnError::InvalidParameter(_)),
            "expected InvalidParameter for version mismatch, got: {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("schema version"),
            "error should mention schema version: {msg}"
        );
    }

    #[test]
    fn missing_schema_version_rejected() {
        let data = make_dataset();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();

        let json = serde_json::to_string(&model).unwrap();

        // Remove the _schema_version field entirely to simulate old payload.
        // serde(default) will set it to 0.
        let stripped = json
            .replace(",\"_schema_version\":1", "")
            .replace("\"_schema_version\":1,", "");

        let loaded: LinearRegression = serde_json::from_str(&stripped).unwrap();
        let err = loaded.predict(&[vec![3.0]]).unwrap_err();
        assert!(
            matches!(err, ScryLearnError::InvalidParameter(_)),
            "expected InvalidParameter for version 0, got: {err}"
        );
    }
}
