//! Testes de integração para funções de layout e cálculos de posição UI.

use neuroscan_core::{
    smoothstep, snfh_pos, navigate_idx, callout_box_w,
    TOP_CASES, menu_entries, dropdown_height,
};

// --- smoothstep ---

#[test]
fn smoothstep_at_zero_is_zero() {
    assert_eq!(smoothstep(0.0), 0.0);
}

#[test]
fn smoothstep_at_one_is_one() {
    assert_eq!(smoothstep(1.0), 1.0);
}

#[test]
fn smoothstep_at_half_is_half() {
    assert!((smoothstep(0.5) - 0.5).abs() < 0.001);
}

#[test]
fn smoothstep_is_monotone() {
    let values: Vec<f32> = (0..=10).map(|i| smoothstep(i as f32 / 10.0)).collect();
    for w in values.windows(2) {
        assert!(w[1] >= w[0], "smoothstep deve ser monotonamente crescente");
    }
}

#[test]
fn smoothstep_clamps_below_zero() {
    assert_eq!(smoothstep(-1.0), 0.0);
    assert_eq!(smoothstep(-99.0), 0.0);
}

#[test]
fn smoothstep_clamps_above_one() {
    assert_eq!(smoothstep(2.0), 1.0);
}

// --- snfh_pos ---

#[test]
fn snfh_pos_at_zero_is_right_side() {
    let w     = 1280.0_f32;
    let box_w = callout_box_w(w);
    let y_ct  = 720.0 * 0.14;
    let box_h = 92.0_f32;
    let (sx, sy) = snfh_pos(0.0, w, box_w, y_ct, box_h);
    let expected_x = w - box_w - 24.0;
    assert!((sx - expected_x).abs() < 0.001, "t=0 deve estar no lado direito");
    assert!((sy - y_ct).abs() < 0.001, "t=0 sy deve ser y_ct");
}

#[test]
fn snfh_pos_at_one_is_left_side() {
    let w     = 1280.0_f32;
    let box_w = callout_box_w(w);
    let y_ct  = 720.0 * 0.14;
    let box_h = 92.0_f32;
    let (sx, sy) = snfh_pos(1.0, w, box_w, y_ct, box_h);
    assert!((sx - 24.0).abs() < 0.001, "t=1 deve estar no lado esquerdo (x=24)");
    assert!((sy - (y_ct + box_h + 12.0)).abs() < 0.001);
}

#[test]
fn snfh_pos_intermediate_is_between_endpoints() {
    let w     = 1280.0_f32;
    let box_w = callout_box_w(w);
    let y_ct  = 720.0 * 0.14;
    let box_h = 92.0_f32;
    let (sx_0, _) = snfh_pos(0.0, w, box_w, y_ct, box_h);
    let (sx_1, _) = snfh_pos(1.0, w, box_w, y_ct, box_h);
    let (sx_m, _) = snfh_pos(0.5, w, box_w, y_ct, box_h);
    assert!(sx_m > sx_1 && sx_m < sx_0,
        "posição intermediária deve estar entre os extremos");
}

// --- navigate_idx ---

#[test]
fn navigate_forward_increments_index() {
    assert_eq!(navigate_idx(0, 1, 10), 1);
    assert_eq!(navigate_idx(5, 1, 10), 6);
}

#[test]
fn navigate_backward_decrements_index() {
    assert_eq!(navigate_idx(5, -1, 10), 4);
    assert_eq!(navigate_idx(1, -1, 10), 0);
}

#[test]
fn navigate_wraps_forward_at_end() {
    let n = TOP_CASES.len();
    assert_eq!(navigate_idx(n - 1, 1, n), 0);
}

#[test]
fn navigate_wraps_backward_at_start() {
    let n = TOP_CASES.len();
    assert_eq!(navigate_idx(0, -1, n), n - 1);
}

// --- callout_box_w ---

#[test]
fn callout_box_w_respects_min() {
    assert_eq!(callout_box_w(100.0), 190.0); // 100 * 0.175 = 17.5 < min 190
}

#[test]
fn callout_box_w_respects_max() {
    assert_eq!(callout_box_w(10000.0), 240.0); // 10000 * 0.175 = 1750 > max 240
}

#[test]
fn callout_box_w_normal_range() {
    let w = callout_box_w(1280.0); // 1280 * 0.175 = 224.0
    assert!((w - 224.0).abs() < 0.001);
}

// --- menu_entries ---

#[test]
fn file_menu_has_three_entries() {
    let entries = menu_entries(0, 0, "0.3.0");
    assert_eq!(entries.len(), 3, "Arquivo: 2 items + 1 separator");
}

#[test]
fn cases_menu_has_correct_count() {
    let entries = menu_entries(1, 0, "0.3.0");
    let expected = 2 + 1 + TOP_CASES.len(); // 2 fixed + 1 sep + 10 cases
    assert_eq!(entries.len(), expected);
}

#[test]
fn cases_menu_marks_current_case() {
    let current = 3_usize;
    let entries = menu_entries(1, current, "0.3.0");
    // Entry at index 3 + current (separator counts as index 2)
    let case_entry_idx = 3 + current; // fixed(2) + sep(1) + current
    let (label, _, is_sep) = &entries[case_entry_idx];
    assert!(!is_sep);
    assert!(label.contains('\u{2713}'), "caso atual deve ter checkmark");
}

#[test]
fn about_menu_contains_version() {
    let version = "0.3.0";
    let entries = menu_entries(2, 0, version);
    assert!(entries[0].0.contains(version), "Sobre deve mostrar a versao");
}

#[test]
fn dropdown_height_increases_with_more_items() {
    // Cases menu has more items than File menu
    let h_file  = dropdown_height(0, 0);
    let h_cases = dropdown_height(1, 0);
    assert!(h_cases > h_file);
}
