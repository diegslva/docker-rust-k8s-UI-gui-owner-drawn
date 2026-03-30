mod camera;
mod mesh;
mod renderer;
mod ui;

use anyhow::{Context, Result};
use camera::OrbitalCamera;
use glam::{Mat4, Vec4};
use mesh::Mesh;
use renderer::{GpuState, MeshEntry, Prim2DBatch};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use ui::{Color, Label};

// ---------------------------------------------------------------------------
// Casos clínicos disponíveis — top 10 por volume tumoral (BraTS 2021)
// ---------------------------------------------------------------------------

const TOP_CASES: &[&str] = &[
    "BRATS_249", "BRATS_141", "BRATS_206", "BRATS_223", "BRATS_155",
    "BRATS_285", "BRATS_020", "BRATS_088", "BRATS_022", "BRATS_117",
];

const CASES_DIR: &str = "assets/models/cases";

// ---------------------------------------------------------------------------
// Metadados do scan
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct ScanMeta {
    case_id:         String,
    dataset:         String,
    modalities:      String,
    et_volume_ml:    f32,
    snfh_volume_ml:  f32,
    netc_volume_ml:  f32,
    total_volume_ml: f32,
}

impl ScanMeta {
    fn load(path: &str) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(t)  => t,
            Err(_) => return Self::default(),
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v)  => v,
            Err(_) => return Self::default(),
        };
        Self {
            case_id:         v["case_id"].as_str().unwrap_or("").to_string(),
            dataset:         v["dataset"].as_str().unwrap_or("").to_string(),
            modalities:      v["modalities"].as_str().unwrap_or("").to_string(),
            et_volume_ml:    v["et_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            snfh_volume_ml:  v["snfh_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            netc_volume_ml:  v["netc_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            total_volume_ml: v["total_volume_ml"].as_f64().unwrap_or(0.0) as f32,
        }
    }

    fn case_path(case_id: &str) -> String {
        format!("{}/{}/scan_meta.json", CASES_DIR, case_id)
    }
}

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Icon, Window, WindowAttributes, WindowId};

const WINDOW_WIDTH:  f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8]  = include_bytes!("../assets/icon_256x256.png");

// --- Cores RGB linear ---
const LH_COLOR:    [f32; 3] = [0.90, 0.72, 0.70];
const RH_COLOR:    [f32; 3] = [0.88, 0.70, 0.68];
const CEREB_COLOR: [f32; 3] = [0.82, 0.62, 0.60];
const INNER_COLOR: [f32; 3] = [0.78, 0.65, 0.72];
const VENTR_COLOR: [f32; 3] = [0.55, 0.72, 0.85];
const ET_COLOR:    [f32; 3] = [0.95, 0.18, 0.12];
const SNFH_COLOR:  [f32; 3] = [0.95, 0.78, 0.05];
const NETC_COLOR:  [f32; 3] = [0.25, 0.50, 0.98];

const LH_ALPHA:    f32 = 0.30;
const RH_ALPHA:    f32 = 0.30;
const CEREB_ALPHA: f32 = 0.55;
const INNER_ALPHA: f32 = 0.80;
const VENTR_ALPHA: f32 = 0.45;

// --- Camera ---
const MOUSE_SENSITIVITY:  f32 = 0.005;
const ZOOM_SENSITIVITY:   f32 = 0.3;
const ZOOM_MIN:           f32 = 1.5;
const ZOOM_MAX:           f32 = 20.0;
const PITCH_LIMIT:        f32 = 1.5;
const AUTO_ROTATE_IDLE_S: f32 = 5.0;   // segundos sem interação para iniciar rotação
const AUTO_ROTATE_SPEED:  f32 = 0.008; // rad/frame a 60fps ≈ 10°/seg

// --- Animações ---
const TRANSITION_DURATION: f32 = 0.45; // segundos para dissolve entre casos
const PULSE_FREQ:           f32 = 1.2;  // Hz do pulso nos pontos de ancoragem

/// Meshes permanentes do cérebro (compartilhadas entre todos os casos)
const BRAIN_DEFS: &[(&str, [f32; 3], f32)] = &[
    (concat!("assets/models/premium/", "Ventricles.obj"),                       VENTR_COLOR, VENTR_ALPHA),
    (concat!("assets/models/premium/", "Thalmus_and_Optic_Tract.obj"),          INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Corpus_Callosum.obj"),                  INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Hippocampus_and_Indusium_Griseum.obj"), INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Putamen_and_Amygdala.obj"),             INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Globus_Pallidus_Externus.obj"),         INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Globus_Pallidus_Internus.obj"),         INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Cerebellum.obj"),                       CEREB_COLOR, CEREB_ALPHA),
    (concat!("assets/models/premium/", "Medulla_and_Pons.obj"),                 CEREB_COLOR, CEREB_ALPHA),
    (concat!("assets/models/premium/", "Left_Cerebral_Hemisphere.obj"),         LH_COLOR,    LH_ALPHA),
    (concat!("assets/models/premium/", "Right_Cerebral_Hemisphere.obj"),        RH_COLOR,    RH_ALPHA),
];

const TUMOR_COUNT: usize = 3;

// ---------------------------------------------------------------------------

struct LoadedMesh {
    mesh:  Mesh,
    tint:  [f32; 3],
    alpha: f32,
}

struct App {
    window:              Option<Arc<Window>>,
    gpu:                 Option<GpuState>,
    /// Índices 0..TUMOR_COUNT = tumores do caso atual; o resto = cérebro permanente
    meshes:              Vec<LoadedMesh>,
    camera:              OrbitalCamera,
    centroids:           [glam::Vec3; TUMOR_COUNT],
    labels_always:       Vec<Label>,
    labels_panel:        Vec<Label>,
    show_panel:          bool,
    scan:                ScanMeta,
    // Navegação entre casos
    current_case:        usize,
    // Animação de transição entre casos
    transition_phase:    f32,   // 0 = idle; 0..1 = animando
    transition_target:   usize,
    transition_loaded:   bool,
    // Microanimações
    last_interaction:    Instant,
    pulse_t:             f32,   // tempo acumulado para o pulso (radianos)
    last_frame:          Instant,
    mouse_pressed:       bool,
    mouse_pos:           Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, gpu: None, meshes: Vec::new(),
            camera: OrbitalCamera::new(4.0),
            centroids: [glam::Vec3::ZERO; TUMOR_COUNT],
            labels_always: Vec::new(),
            labels_panel:  Vec::new(),
            show_panel:    false,
            scan:          ScanMeta::default(),
            current_case:  0,
            transition_phase:  0.0,
            transition_target: 0,
            transition_loaded: false,
            last_interaction:  Instant::now(),
            pulse_t:           0.0,
            last_frame:        Instant::now(),
            mouse_pressed: false,
            mouse_pos: None,
        }
    }

    // -----------------------------------------------------------------------
    // Carregamento de meshes de tumor para o caso atual
    // -----------------------------------------------------------------------

    fn load_tumor_meshes(&mut self, case_id: &str) {
        let case_dir = format!("{}/{}", CASES_DIR, case_id);
        let defs = [
            (format!("{}/tumor_et.obj",   case_dir), ET_COLOR,   1.0f32),
            (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0f32),
            (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0f32),
        ];

        // Carregar num bloco isolado para que o borrow de gpu.device encerre
        // antes de mutar self.meshes
        let new_meshes: Vec<LoadedMesh> = {
            let Some(gpu) = &self.gpu else { return };
            let mut result = Vec::new();
            for (path, tint, alpha) in &defs {
                match Mesh::from_obj(&gpu.device, path) {
                    Ok(m)  => result.push(LoadedMesh { mesh: m, tint: *tint, alpha: *alpha }),
                    Err(e) => warn!(error = %e, path = %path, "mesh de tumor nao encontrada"),
                }
            }
            result
        };

        // Substituir apenas os TUMOR_COUNT primeiros slots
        for (i, lm) in new_meshes.into_iter().enumerate() {
            if i < self.meshes.len() {
                self.meshes[i] = lm;
            }
        }

        // Atualizar centroides
        for i in 0..TUMOR_COUNT {
            self.centroids[i] = self.meshes.get(i)
                .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
        }

        self.scan = ScanMeta::load(&ScanMeta::case_path(case_id));
        info!(case = case_id, total_ml = self.scan.total_volume_ml, "caso carregado");
    }

    /// Inicia a transição animada para o caso no índice `target`.
    fn navigate_case(&mut self, dir: i32) {
        if self.transition_phase > 0.0 { return; }
        let n = TOP_CASES.len();
        let next = ((self.current_case as i32 + dir).rem_euclid(n as i32)) as usize;
        self.transition_target  = next;
        self.transition_phase   = f32::EPSILON; // inicia o ciclo
        self.transition_loaded  = false;
        self.last_interaction   = Instant::now();
    }

    // -----------------------------------------------------------------------
    // Paleta e helpers de UI
    // -----------------------------------------------------------------------

    fn col_header()  -> Color { Color::rgb(226, 232, 240) }
    fn col_dim()     -> Color { Color::rgb( 94, 111, 133) }
    fn col_value()   -> Color { Color::rgb(203, 213, 225) }
    fn col_section() -> Color { Color::rgb( 71,  85, 105) }
    fn col_sep()     -> [f32; 4] { [0.12, 0.17, 0.26, 0.70] }

    fn rgb_f(c: [f32; 3]) -> Color {
        Color::rgb((c[0]*255.0) as u8, (c[1]*255.0) as u8, (c[2]*255.0) as u8)
    }

    fn project_to_screen(p: glam::Vec3, mvp: &[[f32;4];4], w: f32, h: f32) -> Option<(f32, f32)> {
        let m    = Mat4::from_cols_array_2d(mvp);
        let clip = m * Vec4::new(p.x, p.y, p.z, 1.0);
        if clip.w < 0.01 { return None; }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        if nx.abs() > 2.0 || ny.abs() > 2.0 { return None; }
        Some(((nx + 1.0) / 2.0 * w, (1.0 - ny) / 2.0 * h))
    }

    // -----------------------------------------------------------------------
    // Build labels
    // -----------------------------------------------------------------------

    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w  = size.width  as f32;
        let h  = size.height as f32;
        let fs = gpu.font_system_mut();

        let mut always: Vec<Label> = Vec::new();
        let mut panel:  Vec<Label> = Vec::new();

        // ── Título ──────────────────────────────────────────────────────────
        let mut title = Label::new_bold(fs, "NeuroScan", 28.0, Self::col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.04;

        // ── Subtítulo ────────────────────────────────────────────────────────
        let sub_text = if self.scan.case_id.is_empty() {
            "Visualizador Medico 3D  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom  \u{00B7}  I para painel  \u{00B7}  \u{2190}\u{2192} para navegar".to_string()
        } else {
            format!(
                "{}  \u{00B7}  {}  \u{00B7}  {}  \u{00B7}  I para painel  \u{00B7}  \u{2190}\u{2192} para navegar",
                self.scan.case_id, self.scan.dataset, self.scan.modalities
            )
        };
        let mut sub = Label::new(fs, &sub_text, 12.0, Color::rgb(148, 163, 184), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 3.0;

        always.push(title);
        always.push(sub);

        // ── Callouts ─────────────────────────────────────────────────────────
        let y_ct  = h * 0.14;
        let box_w = (w * 0.175).max(190.0).min(240.0);
        let pad   = 12.0_f32;

        // ET — canto superior esquerdo
        {
            let ex = 24.0;
            let ey = y_ct;
            let vol = if self.scan.et_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.et_volume_ml)
            } else { String::new() };
            always.push(Label::new(fs,
                "\u{25CF}  ET  \u{00B7}  Enhancing Tumor",
                10.5, Self::rgb_f(ET_COLOR), ex + pad, ey + 14.0));
            if !vol.is_empty() {
                always.push(Label::new_bold(fs, &vol, 13.0, Color::WHITE, ex + pad, ey + 32.0));
            }
        }

        // SNFH — canto superior direito
        {
            let sx = w - box_w - 24.0;
            let sy = y_ct;
            let vol = if self.scan.snfh_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.snfh_volume_ml)
            } else { String::new() };
            always.push(Label::new(fs,
                "\u{25CF}  SNFH  \u{00B7}  Peritumoral Edema",
                10.5, Self::rgb_f(SNFH_COLOR), sx + pad, sy + 14.0));
            if !vol.is_empty() {
                always.push(Label::new_bold(fs, &vol, 13.0, Color::WHITE, sx + pad, sy + 32.0));
            }
        }

        // NETC — centro inferior
        {
            let nx = (w / 2.0 - box_w / 2.0).max(24.0);
            let ny = (h - 155.0).max(y_ct + 64.0 + 20.0);
            let vol = if self.scan.netc_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.netc_volume_ml)
            } else { String::new() };
            always.push(Label::new(fs,
                "\u{25CF}  NETC  \u{00B7}  Necrotic Core",
                10.5, Self::rgb_f(NETC_COLOR), nx + pad, ny + 14.0));
            if !vol.is_empty() {
                always.push(Label::new_bold(fs, &vol, 13.0, Color::WHITE, nx + pad, ny + 32.0));
            }
        }

        // ── Indicador de caso (centro inferior) ──────────────────────────────
        {
            let idx  = self.current_case + 1;
            let n    = TOP_CASES.len();
            let text = format!("\u{2190}   Caso {}  /  {}   \u{2192}", idx, n);
            let mut nav = Label::new(fs, &text, 11.0, Color::rgb(100, 116, 139), 0.0, 0.0);
            nav.x = (w - nav.measured_width()) / 2.0;
            nav.y = h - 22.0;
            always.push(nav);
        }

        // ── Legenda inferior esquerda ─────────────────────────────────────────
        let leg_font = 11.0;
        let leg_gap  = leg_font * 1.8;
        let leg_items = [
            (ET_COLOR,   "ET",   self.scan.et_volume_ml),
            (SNFH_COLOR, "SNFH", self.scan.snfh_volume_ml),
            (NETC_COLOR, "NETC", self.scan.netc_volume_ml),
        ];
        let n_leg = leg_items.len() as f32;
        let leg_y = h - n_leg * leg_gap - 30.0;
        for (i, (rgb, name, vol)) in leg_items.iter().enumerate() {
            let text = if *vol > 0.0 {
                format!("\u{25CF}  {}  {:.1} mL", name, vol)
            } else {
                format!("\u{25CF}  {}", name)
            };
            always.push(Label::new(fs, &text, leg_font, Self::rgb_f(*rgb),
                28.0, leg_y + i as f32 * leg_gap));
        }

        // ── Painel clínico (toggle I) ─────────────────────────────────────────
        let px = (w - 268.0).max(w * 0.72);
        let pl = px + 14.0;
        let mut py = h * 0.14;

        macro_rules! section {
            ($text:expr) => {{
                panel.push(Label::new(fs, $text, 9.5, Self::col_section(), pl, py));
                py += 18.0;
            }};
        }
        macro_rules! kv {
            ($key:expr, $val:expr) => {{
                panel.push(Label::new(fs, $key, 10.0, Self::col_dim(), pl, py));
                panel.push(Label::new(fs, $val, 11.0, Self::col_value(), pl + 90.0, py));
                py += 16.0;
            }};
        }
        macro_rules! line_text {
            ($text:expr, $col:expr, $size:expr) => {{
                panel.push(Label::new(fs, $text, $size, $col, pl, py));
                py += $size * 1.5;
            }};
        }

        let case_val     = if self.scan.case_id.is_empty() { "—".to_string() } else { self.scan.case_id.clone() };
        let dataset_val  = if self.scan.dataset.is_empty() { "—".to_string() } else { self.scan.dataset.clone() };
        let nav_val      = format!("{} / {}", self.current_case + 1, TOP_CASES.len());

        section!("ANALISE VOLUMETRICA");
        kv!("Caso", &case_val);
        kv!("Navegacao", &nav_val);
        kv!("Protocolo", &dataset_val);
        if !self.scan.modalities.is_empty() {
            panel.push(Label::new(fs, "Modalidades", 10.0, Self::col_dim(), pl, py));
            panel.push(Label::new(fs, "FLAIR · T1w", 11.0, Self::col_value(), pl + 90.0, py));
            py += 14.0;
            panel.push(Label::new(fs, "T1ce · T2w",  11.0, Self::col_value(), pl + 90.0, py));
            py += 16.0;
        }
        kv!("Metodo",   "nnUNet 2D Slice");
        kv!("Acuracia", "Dice  0.865");
        py += 8.0;

        section!("CLASSIFICACAO TUMORAL");
        line_text!("WHO 2021  ·  Grau IV",        Self::col_value(), 11.0);
        line_text!("Glioblastoma Multiforme",      Self::col_header(), 11.5);
        py += 8.0;

        section!("VOLUMES SEGMENTADOS");
        let et_s   = format!("{:.1} mL", self.scan.et_volume_ml);
        let snfh_s = format!("{:.1} mL", self.scan.snfh_volume_ml);
        let netc_s = format!("{:.1} mL", self.scan.netc_volume_ml);
        let tot_s  = format!("{:.1} mL", self.scan.total_volume_ml);
        panel.push(Label::new(fs, "\u{25CF}  ET    Realce tumoral",    10.5, Self::rgb_f(ET_COLOR),   pl, py));
        panel.push(Label::new(fs, &et_s,   11.5, Color::WHITE, pl + 168.0, py)); py += 15.0;
        panel.push(Label::new(fs, "\u{25CF}  SNFH  Edema peritumoral", 10.5, Self::rgb_f(SNFH_COLOR), pl, py));
        panel.push(Label::new(fs, &snfh_s, 11.5, Color::WHITE, pl + 168.0, py)); py += 15.0;
        panel.push(Label::new(fs, "\u{25CF}  NETC  Nucleo necrotico",  10.5, Self::rgb_f(NETC_COLOR), pl, py));
        panel.push(Label::new(fs, &netc_s, 11.5, Color::WHITE, pl + 168.0, py)); py += 18.0;
        panel.push(Label::new(fs, "Total", 10.0, Self::col_dim(), pl + 120.0, py));
        panel.push(Label::new_bold(fs, &tot_s, 13.0, Color::WHITE, pl + 168.0, py));
        py += 22.0;

        section!("METODOLOGIA");
        for line in &[
            "Segmentacao automatica por rede",
            "neural convolucional 2D treinada",
            "em 484 casos (BraTS 2021).",
            "Inferencia slice-a-slice com 4",
            "canais MRI simultaneos.",
        ] {
            panel.push(Label::new(fs, line, 10.0, Self::col_dim(), pl, py));
            py += 14.0;
        }

        self.labels_always = always;
        self.labels_panel  = panel;
    }

    // -----------------------------------------------------------------------
    // Primitivas 2D
    // -----------------------------------------------------------------------

    fn build_primitives(
        &self,
        mvp:       &[[f32; 4]; 4],
        w:          f32,
        h:          f32,
        pulse_t:    f32,
        transition: f32,
    ) -> Prim2DBatch {
        let mut b    = Prim2DBatch::new();
        let y_ct     = h * 0.14;
        let box_w    = (w * 0.175).max(190.0).min(240.0);
        let box_h    = 64.0_f32;
        let bg       = [0.04, 0.06, 0.11, 0.88_f32];
        // Pulso: raio do ponto de ancoragem oscila suavemente
        let dot_r    = 3.5 + 1.5 * (pulse_t * std::f32::consts::TAU * PULSE_FREQ).sin().abs();

        // ET callout (top-left)
        let et_x = 24.0;
        let et_y = y_ct;
        b.rect(et_x, et_y, box_w, box_h, bg, w, h);
        b.rect(et_x, et_y, box_w, 2.0, [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 1.0], w, h);
        if let Some((cx, cy)) = Self::project_to_screen(self.centroids[0], mvp, w, h) {
            b.line(et_x + box_w, et_y + box_h * 0.5, cx, cy,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.60], 1.5, w, h);
            b.rect(cx - dot_r, cy - dot_r, dot_r*2.0, dot_r*2.0,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.90], w, h);
        }

        // SNFH callout (top-right)
        let snfh_x = w - box_w - 24.0;
        let snfh_y = y_ct;
        b.rect(snfh_x, snfh_y, box_w, box_h, bg, w, h);
        b.rect(snfh_x, snfh_y, box_w, 2.0, [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 1.0], w, h);
        if let Some((cx, cy)) = Self::project_to_screen(self.centroids[1], mvp, w, h) {
            b.line(snfh_x, snfh_y + box_h * 0.5, cx, cy,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.60], 1.5, w, h);
            b.rect(cx - dot_r, cy - dot_r, dot_r*2.0, dot_r*2.0,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.90], w, h);
        }

        // NETC callout (bottom-center)
        let netc_x = (w / 2.0 - box_w / 2.0).max(24.0);
        let netc_y = (h - 155.0).max(y_ct + box_h + 20.0);
        b.rect(netc_x, netc_y, box_w, box_h, bg, w, h);
        b.rect(netc_x, netc_y, box_w, 2.0, [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 1.0], w, h);
        if let Some((cx, cy)) = Self::project_to_screen(self.centroids[2], mvp, w, h) {
            b.line(netc_x + box_w * 0.5, netc_y, cx, cy,
                [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 0.60], 1.5, w, h);
            b.rect(cx - dot_r, cy - dot_r, dot_r*2.0, dot_r*2.0,
                [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 0.90], w, h);
        }

        // Fundo do painel clínico
        if self.show_panel {
            let px = (w - 268.0).max(w * 0.72);
            let py = h * 0.14 - 8.0;
            let pw = 258.0_f32;
            let ph = h * 0.80;
            b.rect(px, py, pw, ph, [0.03, 0.05, 0.09, 0.90], w, h);
            b.rect(px, py, 2.0, ph, [0.35, 0.55, 0.85, 0.40], w, h);
            // Separadores
            for sep_y in &[
                h * 0.14 + 82.0,
                h * 0.14 + 136.0,
                h * 0.14 + 214.0,
                h * 0.14 + 242.0,
            ] {
                b.rect(px + 10.0, *sep_y, pw - 20.0, 1.0, Self::col_sep(), w, h);
            }
        }

        // Overlay de transição — dissolve sinusoidal suave
        if transition > 0.0 {
            let alpha = (transition * std::f32::consts::PI).sin() * 0.92;
            b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, alpha], w, h);
        }

        b
    }
}

// ---------------------------------------------------------------------------

fn load_embedded_icon() -> Result<Icon> {
    let img = image::load_from_memory(ICON_BYTES)
        .context("falha ao decodificar icone")?.into_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).context("falha ao criar Icon")
}

fn center_position(event_loop: &ActiveEventLoop) -> Option<PhysicalPosition<i32>> {
    let monitor = event_loop.primary_monitor()
        .or_else(|| event_loop.available_monitors().next())?;
    let ms = monitor.size();
    let mp = monitor.position();
    let scale = monitor.scale_factor();
    let pw = (WINDOW_WIDTH  * scale) as i32;
    let ph = (WINDOW_HEIGHT * scale) as i32;
    Some(PhysicalPosition::new(mp.x + (ms.width as i32 - pw) / 2, mp.y + (ms.height as i32 - ph) / 2))
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let icon = load_embedded_icon().ok();
        let mut attrs = WindowAttributes::default()
            .with_title("NeuroScan — Visualizador Medico 3D")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        if let Some(ic) = icon { attrs = attrs.with_window_icon(Some(ic)); }

        let window = match event_loop.create_window(attrs) {
            Ok(w)  => Arc::new(w),
            Err(e) => { warn!(error = %e, "falha ao criar janela"); event_loop.exit(); return; }
        };
        if let Some(pos) = center_position(event_loop) { window.set_outer_position(pos); }

        let gpu = match pollster::block_on(GpuState::new(Arc::clone(&window))) {
            Ok(s)  => s,
            Err(e) => { warn!(error = %e, "falha ao inicializar wgpu"); event_loop.exit(); return; }
        };
        self.gpu = Some(gpu);

        // Carregar cérebro permanente
        {
            let device = &self.gpu.as_ref().unwrap().device;
            for (path, tint, alpha) in BRAIN_DEFS {
                match Mesh::from_obj(device, path) {
                    Ok(m)  => self.meshes.push(LoadedMesh { mesh: m, tint: *tint, alpha: *alpha }),
                    Err(e) => warn!(error = %e, path, "mesh cerebro nao encontrada"),
                }
            }
        }

        // Pré-alocar slots para os tumores no início do vetor
        // (inserimos 3 placeholders e depois substituímos via load_tumor_meshes)
        for _ in 0..TUMOR_COUNT {
            self.meshes.insert(0, LoadedMesh {
                mesh:  Mesh::from_obj(&self.gpu.as_ref().unwrap().device,
                    &format!("{}/{}/tumor_et.obj", CASES_DIR, TOP_CASES[0]))
                    .unwrap_or_else(|_| panic!("caso inicial nao encontrado: {}", TOP_CASES[0])),
                tint:  [1.0; 3],
                alpha: 1.0,
            });
        }

        self.load_tumor_meshes(TOP_CASES[0]);
        info!(total_meshes = self.meshes.len(), "scene carregada");

        self.camera.target   = glam::Vec3::ZERO;
        self.camera.distance = 4.0;

        let size = window.inner_size();
        self.build_labels(size);
        self.last_frame       = Instant::now();
        self.last_interaction = Instant::now();
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => { info!("close"); event_loop.exit(); }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.last_interaction = Instant::now();
                    match &event.logical_key {
                        Key::Character(ch) => match ch.as_str() {
                            "i" | "I" => { self.show_panel = !self.show_panel; }
                            _ => {}
                        },
                        Key::Named(NamedKey::ArrowRight) => self.navigate_case( 1),
                        Key::Named(NamedKey::ArrowLeft)  => self.navigate_case(-1),
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    self.last_interaction = Instant::now();
                    if let Some(w) = &self.window {
                        w.set_cursor(if self.mouse_pressed { CursorIcon::Grabbing } else { CursorIcon::Grab });
                    }
                    if state == ElementState::Released { self.mouse_pos = None; }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    self.last_interaction = Instant::now();
                    if let Some((px, py)) = self.mouse_pos {
                        self.camera.yaw   += (position.x - px) as f32 * MOUSE_SENSITIVITY;
                        self.camera.pitch  = (self.camera.pitch - (position.y - py) as f32 * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((position.x, position.y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.last_interaction = Instant::now();
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.01,
                };
                self.camera.distance = (self.camera.distance - scroll * ZOOM_SENSITIVITY)
                    .clamp(ZOOM_MIN, ZOOM_MAX);
            }

            WindowEvent::Resized(new_size) => {
                debug!(w = new_size.width, h = new_size.height, "resize");
                if let Some(gpu) = &mut self.gpu { gpu.resize(new_size); }
                self.build_labels(new_size);
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32().min(0.05);
                self.last_frame = Instant::now();

                // ── Microanimação: rotação automática após inatividade ──────
                if self.last_interaction.elapsed().as_secs_f32() > AUTO_ROTATE_IDLE_S {
                    self.camera.yaw += AUTO_ROTATE_SPEED * dt * 60.0;
                }

                // ── Pulso dos pontos de ancoragem ──────────────────────────
                self.pulse_t += dt;

                // ── Transição de caso ──────────────────────────────────────
                let mut rebuild_needed = false;
                if self.transition_phase > 0.0 {
                    self.transition_phase += dt / TRANSITION_DURATION;

                    // No meio da transição: trocar o caso
                    if self.transition_phase >= 0.5 && !self.transition_loaded {
                        self.transition_loaded = true;
                        self.current_case = self.transition_target;
                        self.load_tumor_meshes(TOP_CASES[self.current_case]);
                        rebuild_needed = true;
                    }

                    if self.transition_phase >= 1.0 {
                        self.transition_phase = 0.0;
                        self.transition_loaded = false;
                    }
                }

                let size = PhysicalSize::new(
                    self.gpu.as_ref().map_or(1280, |g| g.config.width),
                    self.gpu.as_ref().map_or(720,  |g| g.config.height),
                );
                if rebuild_needed { self.build_labels(size); }

                let cam   = self.camera.build_uniform(size.width, size.height);
                let prims = self.build_primitives(&cam.mvp, size.width as f32, size.height as f32,
                    self.pulse_t, self.transition_phase);

                let mut label_refs: Vec<&Label> = self.labels_always.iter().collect();
                if self.show_panel { label_refs.extend(self.labels_panel.iter()); }

                if let Some(gpu) = &mut self.gpu {
                    let entries: Vec<MeshEntry> = self.meshes.iter()
                        .map(|m| MeshEntry { mesh: &m.mesh, tint: m.tint, alpha: m.alpha })
                        .collect();
                    if let Err(e) = gpu.render(&cam, &entries, &label_refs, &prims) {
                        warn!(error = %e, "erro no render");
                    }
                }

                if let Some(w) = &self.window { w.request_redraw(); }
            }

            _ => {}
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) { info!("shutdown"); }
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt::Subscriber::builder().with_env_filter(filter).with_target(true).with_thread_ids(true).init();
    info!(version = env!("CARGO_PKG_VERSION"), "NeuroScan viewer iniciando");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
