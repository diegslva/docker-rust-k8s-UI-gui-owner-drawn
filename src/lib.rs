//! NeuroScan Core — lógica pura testável (sem dependência de wgpu/winit).
//!
//! Autor: Diego L. Silva (github.com/diegslva)

// ---------------------------------------------------------------------------
// Casos clínicos
// ---------------------------------------------------------------------------

pub const TOP_CASES: &[&str] = &[
    "BRATS_249",
    "BRATS_141",
    "BRATS_206",
    "BRATS_223",
    "BRATS_155",
    "BRATS_285",
    "BRATS_020",
    "BRATS_088",
    "BRATS_022",
    "BRATS_117",
];

pub const CASES_DIR: &str = "assets/models/cases";

// ---------------------------------------------------------------------------
// ScanMeta
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ScanMeta {
    pub case_id: String,
    pub dataset: String,
    pub modalities: String,
    pub et_volume_ml: f32,
    pub snfh_volume_ml: f32,
    pub netc_volume_ml: f32,
    pub total_volume_ml: f32,
}

impl ScanMeta {
    pub fn load(path: &str) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return Self::default(),
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Self::default(),
        };
        Self {
            case_id: v["case_id"].as_str().unwrap_or("").to_string(),
            dataset: v["dataset"].as_str().unwrap_or("").to_string(),
            modalities: v["modalities"].as_str().unwrap_or("").to_string(),
            et_volume_ml: v["et_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            snfh_volume_ml: v["snfh_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            netc_volume_ml: v["netc_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            total_volume_ml: v["total_volume_ml"].as_f64().unwrap_or(0.0) as f32,
        }
    }

    pub fn case_path(case_id: &str) -> String {
        format!("{}/{}/scan_meta.json", CASES_DIR, case_id)
    }
}

// ---------------------------------------------------------------------------
// Layout — funções puras para cálculos de posição UI
// ---------------------------------------------------------------------------

/// Smoothstep easing: começa devagar, acelera, termina devagar.
/// Garante: smoothstep(0.0) = 0.0, smoothstep(1.0) = 1.0.
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Posição interpolada do callout SNFH durante a animação de slide.
///
/// `anim_t = 0.0` → posição direita (painel fechado)
/// `anim_t = 1.0` → posição esquerda/abaixo-ET (painel aberto)
pub fn snfh_pos(anim_t: f32, w: f32, box_w: f32, y_ct: f32, box_h: f32) -> (f32, f32) {
    let ease = smoothstep(anim_t);
    let right_x = w - box_w - 24.0;
    let left_x = 24.0_f32;
    let sx = right_x + (left_x - right_x) * ease;
    let sy = y_ct + (box_h + 12.0) * ease;
    (sx, sy)
}

/// Calcula o índice do próximo caso com wrap-around circular.
pub fn navigate_idx(current: usize, dir: i32, n: usize) -> usize {
    ((current as i32 + dir).rem_euclid(n as i32)) as usize
}

/// Largura de um callout box em pixels dado o viewport width.
pub fn callout_box_w(viewport_w: f32) -> f32 {
    (viewport_w * 0.175).clamp(190.0, 240.0)
}

// ---------------------------------------------------------------------------
// Menu — entradas dos menus como dados puros
// ---------------------------------------------------------------------------

/// Menu bar constants (espelham os definidos em main.rs).
pub const MENU_BAR_H: f32 = 26.0;
pub const MENU_ITEM_H: f32 = 22.0;
pub const MENU_SEP_H: f32 = 9.0;
pub const MENU_DROP_W: f32 = 220.0;
pub const MENU_TOP_XS: [f32; 3] = [8.0, 78.0, 140.0];
pub const MENU_TOP_WS: [f32; 3] = [68.0, 58.0, 58.0];

/// Retorna as entradas do menu `menu_id` como (texto, shortcut, is_separator).
pub fn menu_entries(
    menu_id: i32,
    current_case: usize,
    pkg_version: &str,
) -> Vec<(String, String, bool)> {
    match menu_id {
        0 => vec![
            ("Abrir Volume NIfTI...".into(), "O".into(), false),
            (String::new(), String::new(), true),
            ("Sair".into(), String::new(), false),
        ],
        1 => {
            let mut v: Vec<(String, String, bool)> = vec![
                ("Caso Anterior".into(), "\u{2190}".into(), false),
                ("Próximo Caso".into(), "\u{2192}".into(), false),
                (String::new(), String::new(), true),
            ];
            for (i, id) in TOP_CASES.iter().enumerate() {
                let label = if i == current_case {
                    format!("\u{2713}  {}", id)
                } else {
                    format!("    {}", id)
                };
                v.push((label, String::new(), false));
            }
            v
        }
        2 => vec![
            (format!("NeuroScan  v{}", pkg_version), String::new(), false),
            (
                "nnUNet 2D  \u{00B7}  BraTS 2021 + 2023".into(),
                String::new(),
                false,
            ),
            (
                "Dice 0.822  \u{00B7}  1.735 casos".into(),
                String::new(),
                false,
            ),
            (String::new(), String::new(), true),
            (
                "Diego L. Silva  \u{00B7}  github.com/diegslva".into(),
                String::new(),
                false,
            ),
        ],
        _ => vec![],
    }
}

/// Altura total do dropdown para o menu `menu_id`.
pub fn dropdown_height(menu_id: i32, current_case: usize) -> f32 {
    menu_entries(menu_id, current_case, "0.0.0")
        .iter()
        .map(|(_, _, sep)| if *sep { MENU_SEP_H } else { MENU_ITEM_H })
        .sum::<f32>()
        + 4.0
}
