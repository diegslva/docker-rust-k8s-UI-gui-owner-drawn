//! Testes de integração para ScanMeta — parsing, defaults e caminhos.

use neuroscan_core::{ScanMeta, CASES_DIR};

#[test]
fn scan_meta_default_has_empty_fields() {
    let meta = ScanMeta::default();
    assert!(meta.case_id.is_empty());
    assert!(meta.dataset.is_empty());
    assert_eq!(meta.et_volume_ml, 0.0);
    assert_eq!(meta.snfh_volume_ml, 0.0);
    assert_eq!(meta.netc_volume_ml, 0.0);
    assert_eq!(meta.total_volume_ml, 0.0);
}

#[test]
fn scan_meta_load_nonexistent_returns_default() {
    let meta = ScanMeta::load("/nonexistent/path/scan_meta.json");
    assert!(meta.case_id.is_empty());
    assert_eq!(meta.et_volume_ml, 0.0);
}

#[test]
fn scan_meta_load_invalid_json_returns_default() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan_meta.json");
    std::fs::write(&path, b"{ invalid json }").unwrap();
    let meta = ScanMeta::load(path.to_str().unwrap());
    assert!(meta.case_id.is_empty());
}

#[test]
fn scan_meta_load_valid_json() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan_meta.json");
    let json = r#"{
        "case_id": "BRATS_249",
        "dataset": "BraTS2021",
        "modalities": "FLAIR/T1w/T1ce/T2w",
        "et_volume_ml": 12.5,
        "snfh_volume_ml": 48.3,
        "netc_volume_ml": 7.1,
        "total_volume_ml": 67.9
    }"#;
    std::fs::write(&path, json.as_bytes()).unwrap();
    let meta = ScanMeta::load(path.to_str().unwrap());
    assert_eq!(meta.case_id, "BRATS_249");
    assert_eq!(meta.dataset, "BraTS2021");
    assert!((meta.et_volume_ml   - 12.5).abs() < 0.001);
    assert!((meta.snfh_volume_ml - 48.3).abs() < 0.001);
    assert!((meta.total_volume_ml - 67.9).abs() < 0.001);
}

#[test]
fn scan_meta_load_partial_json_fills_defaults() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan_meta.json");
    // Only case_id present — volumes should default to 0.0
    let json = r#"{ "case_id": "BRATS_001" }"#;
    std::fs::write(&path, json.as_bytes()).unwrap();
    let meta = ScanMeta::load(path.to_str().unwrap());
    assert_eq!(meta.case_id, "BRATS_001");
    assert_eq!(meta.et_volume_ml, 0.0);
}

#[test]
fn case_path_format_is_correct() {
    let path = ScanMeta::case_path("BRATS_249");
    assert_eq!(path, format!("{}/BRATS_249/scan_meta.json", CASES_DIR));
}
