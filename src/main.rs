mod camera;
mod mesh;
mod renderer;
mod ui;

use anyhow::{Context, Result};
use camera::OrbitalCamera;
use glam::{Mat4, Vec4};
use mesh::Mesh;
use neuroscan_core::{ScanMeta, TOP_CASES, CASES_DIR};
use renderer::{GpuState, MeshEntry, Prim2DBatch};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use ui::{Color, Label};

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
const TRANSITION_DURATION:    f32 = 0.45; // segundos para dissolve entre casos
const PULSE_FREQ:              f32 = 1.2;  // Hz do pulso nos pontos de ancoragem
const SPLASH_FADEOUT_DURATION: f32 = 0.70; // segundos para fade-out da splash

// --- Menu bar owner-drawn ---
const MENU_BAR_H:  f32 = 26.0;
const MENU_ITEM_H: f32 = 22.0;
const MENU_SEP_H:  f32 = 9.0;
const MENU_DROP_W: f32 = 220.0;
/// X de início e largura de cada item da barra: Arquivo | Casos | Sobre
const MENU_TOP_XS: [f32; 3] = [8.0, 78.0, 140.0];
const MENU_TOP_WS: [f32; 3] = [68.0, 58.0, 58.0];

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
    labels_snfh:         Vec<Label>,  // callout SNFH — posição mutada diretamente (sem re-shaping)
    labels_menu:         Vec<Label>,  // dropdown do menu ativo
    show_panel:          bool,
    // Menu bar
    menu_open:           i32,  // -1 = fechado; 0=Arquivo, 1=Casos, 2=Sobre
    menu_hover_top:      i32,  // -1=nenhum, 0..2 = top-item sob cursor
    menu_hover_item:     i32,  // -1=nenhum, índice no dropdown atual
    scan:                ScanMeta,
    // Janela
    window_shown:  bool,
    // Splash screen
    splash_done:   bool,
    splash_t:      f32,
    splash_fade:   f32,   // 0..1 durante fade-out
    splash_rx:     Option<mpsc::Receiver<Vec<LoadedMesh>>>,
    splash_labels: Vec<Label>,
    // Inferência real — file picker + subprocess Python
    dialog_rx:     Option<mpsc::Receiver<std::path::PathBuf>>,
    infer_active:  bool,
    infer_rx:      Option<mpsc::Receiver<bool>>, // true = ok, false = erro
    // Navegação entre casos
    current_case:        usize,
    // Animação de transição entre casos
    transition_phase:    f32,   // 0 = idle; 0..0.5 = fade-in; 0.5..1.0 = fade-out
    transition_target:   usize,
    loading_rx:          Option<mpsc::Receiver<Vec<LoadedMesh>>>,
    // Microanimações
    last_interaction:    Instant,
    pulse_t:             f32,   // tempo acumulado para o pulso (radianos)
    spinner_angle:       f32,   // ângulo atual da cabeça do arc spinner
    snfh_anim_t:         f32,   // 0.0 = direita (painel fechado), 1.0 = esquerda (painel aberto)
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
            labels_always:   Vec::new(),
            labels_panel:    Vec::new(),
            labels_snfh:     Vec::new(),
            labels_menu:     Vec::new(),
            show_panel:      false,
            menu_open:       -1,
            menu_hover_top:  -1,
            menu_hover_item: -1,
            scan:          ScanMeta::default(),
            window_shown:  false,
            splash_done:   false,
            splash_t:      0.0,
            splash_fade:   0.0,
            splash_rx:     None,
            splash_labels: Vec::new(),
            dialog_rx:     None,
            infer_active:  false,
            infer_rx:      None,
            current_case:  0,
            transition_phase:  0.0,
            transition_target: 0,
            loading_rx:        None,
            last_interaction:  Instant::now(),
            pulse_t:           0.0,
            spinner_angle:     0.0,
            snfh_anim_t:       0.0,
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
    /// O carregamento das meshes acontece em thread de fundo — sem travar o render loop.
    fn navigate_case(&mut self, dir: i32) {
        if self.transition_phase > 0.0 { return; }
        let n    = TOP_CASES.len();
        let next = ((self.current_case as i32 + dir).rem_euclid(n as i32)) as usize;
        self.transition_target = next;
        self.transition_phase  = f32::EPSILON;
        self.spinner_angle     = 0.0;
        self.last_interaction  = Instant::now();

        // Clonar o device (Arc interno do wgpu — zero custo) para a thread
        let device   = self.gpu.as_ref().unwrap().device.clone();
        let case_id  = TOP_CASES[next].to_string();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let case_dir = format!("{}/{}", CASES_DIR, case_id);
            let defs = [
                (format!("{}/tumor_et.obj",   case_dir), ET_COLOR,   1.0_f32),
                (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0_f32),
                (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0_f32),
            ];
            let meshes: Vec<LoadedMesh> = defs.iter()
                .filter_map(|(path, tint, alpha)| {
                    Mesh::from_obj(&device, path).ok()
                        .map(|m| LoadedMesh { mesh: m, tint: *tint, alpha: *alpha })
                })
                .collect();
            let _ = tx.send(meshes);
        });

        self.loading_rx = Some(rx);
    }

    // -----------------------------------------------------------------------
    // Paleta e helpers de UI
    // -----------------------------------------------------------------------

    fn col_header()  -> Color { Color::rgb(226, 232, 240) }
    fn col_dim()     -> Color { Color::rgb( 94, 111, 133) }
    fn col_value()   -> Color { Color::rgb(203, 213, 225) }
    fn col_section() -> Color { Color::rgb( 71,  85, 105) }
    fn col_sep()     -> [f32; 4] { [0.12, 0.17, 0.26, 0.70] }

    // -----------------------------------------------------------------------
    // Splash screen — labels e primitivas
    // -----------------------------------------------------------------------

    fn build_splash_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w  = size.width  as f32;
        let h  = size.height as f32;
        let cy = h / 2.0 + 28.0;  // centro da animação, levemente abaixo do meio
        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(fs, "NeuroScan", 50.0, Color::rgb(222, 234, 248), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = cy - 122.0;

        let mut sub = Label::new(fs, "Visualizador Medico 3D", 15.5, Color::rgb(88, 128, 168), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 5.0;

        self.splash_labels = vec![title, sub];
    }

    /// Geometria animada da splash: fundo escuro, scan-line MRI, anéis pulsantes, arco orbital.
    fn build_splash_primitives(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b  = Prim2DBatch::new();
        let cx     = w / 2.0;
        let cy     = h / 2.0 + 28.0;
        let t      = self.splash_t;
        let orbit  = self.spinner_angle;

        // Fundo
        b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, 1.0], w, h);

        // Linha de varredura horizontal — estilo scanner MRI
        let scan_y = ((t * 0.36).fract() * h).clamp(0.0, h - 2.0);
        b.rect(0.0, scan_y - 1.0, w, 2.5, [0.30, 0.58, 0.88, 0.13], w, h);
        b.rect(w * 0.25, scan_y - 0.5, w * 0.50, 1.0, [0.52, 0.78, 1.0, 0.20], w, h);

        // Três anéis pulsantes (defasados em 1/3 do ciclo)
        let ring_n     = 24_usize;
        let ring_r_max = 88.0_f32;
        let cycle      = 2.8_f32;
        for i in 0..3usize {
            let phase = ((t / cycle) + (i as f32 / 3.0)).fract();
            let r     = phase * ring_r_max;
            let alpha = (1.0 - phase).powf(1.8) * 0.30;
            if alpha < 0.01 { continue; }
            for j in 0..ring_n {
                let a0 = (j     as f32 / ring_n as f32) * std::f32::consts::TAU;
                let a1 = ((j+1) as f32 / ring_n as f32) * std::f32::consts::TAU;
                b.line(cx + r * a0.cos(), cy + r * a0.sin(),
                       cx + r * a1.cos(), cy + r * a1.sin(),
                       [0.22, 0.52, 0.86, alpha], 1.2, w, h);
            }
        }

        // Trilha circular do arco orbital
        let orbit_r = 44.0_f32;
        for j in 0..36usize {
            let a0 = (j     as f32 / 36.0) * std::f32::consts::TAU;
            let a1 = ((j+1) as f32 / 36.0) * std::f32::consts::TAU;
            b.line(cx + orbit_r * a0.cos(), cy + orbit_r * a0.sin(),
                   cx + orbit_r * a1.cos(), cy + orbit_r * a1.sin(),
                   [0.12, 0.20, 0.36, 0.26], 1.0, w, h);
        }

        // Arco orbital com cauda gradiente
        let arc_n    = 28_usize;
        let arc_span = std::f32::consts::TAU * (260.0 / 360.0);
        let tail_a   = orbit - arc_span;
        for j in 0..arc_n {
            let t0 = j     as f32 / arc_n as f32;
            let t1 = (j+1) as f32 / arc_n as f32;
            let a0 = tail_a + t0 * arc_span;
            let a1 = tail_a + t1 * arc_span;
            let br = t0.powf(1.3);
            b.line(cx + orbit_r * a0.cos(), cy + orbit_r * a0.sin(),
                   cx + orbit_r * a1.cos(), cy + orbit_r * a1.sin(),
                   [0.42 + 0.22 * br, 0.65 + 0.16 * br, 1.0, br * 0.88],
                   2.0, w, h);
        }

        // Ponto brilhante na cabeça do arco
        let hx = cx + orbit_r * orbit.cos();
        let hy = cy + orbit_r * orbit.sin();
        let hr = 2.6_f32;
        b.rect(hx - hr, hy - hr, hr * 2.0, hr * 2.0, [0.70, 0.90, 1.0, 0.92], w, h);

        b
    }

    // -----------------------------------------------------------------------
    // Menu bar — entradas, rebuild de labels do dropdown, posições SNFH
    // -----------------------------------------------------------------------

    /// Retorna as entradas do menu `menu_id` como (texto, shortcut_hint, is_separator).
    fn build_menu_entries(&self, menu_id: i32) -> Vec<(String, String, bool)> {
        neuroscan_core::menu_entries(menu_id, self.current_case, env!("CARGO_PKG_VERSION"))
    }

    /// Altura total do dropdown para o menu `menu_id`.
    fn dropdown_height(&self, menu_id: i32) -> f32 {
        neuroscan_core::dropdown_height(menu_id, self.current_case)
    }

    /// Constrói `labels_menu` para o dropdown do menu atualmente aberto.
    /// Chamado quando `menu_open` muda.
    fn rebuild_menu_labels(&mut self, size: PhysicalSize<u32>) {
        if self.menu_open < 0 {
            self.labels_menu = Vec::new();
            return;
        }
        let menu_id = self.menu_open;
        let entries  = self.build_menu_entries(menu_id);
        let top_x    = MENU_TOP_XS[menu_id as usize];
        let text_x   = top_x + 12.0;
        let right_x  = top_x + MENU_DROP_W - 14.0;
        let w = size.width as f32;
        let h = size.height as f32;

        let Some(gpu) = &mut self.gpu else { return };
        let fs = gpu.font_system_mut();
        let col_item  = Color::rgb(188, 200, 218);
        let col_sc    = Color::rgb(100, 116, 139);
        // ✓ item (current case) em verde suave
        let col_check = Color::rgb(100, 200, 140);

        let mut labels = Vec::new();
        let mut y = MENU_BAR_H + 2.0;

        for (label, shortcut, is_sep) in &entries {
            if *is_sep { y += MENU_SEP_H; continue; }
            let col = if label.starts_with('\u{2713}') { col_check } else { col_item };
            labels.push(Label::new(fs, label, 11.0, col, text_x, y + 4.0));
            if !shortcut.is_empty() {
                let mut sc = Label::new(fs, shortcut, 11.0, col_sc, 0.0, y + 4.0);
                sc.x = right_x - sc.measured_width();
                labels.push(sc);
            }
            y += MENU_ITEM_H;
        }
        let _ = (w, h); // apenas evita warnings de unused
        self.labels_menu = labels;
    }

    /// Atualiza `.x`/`.y` dos labels SNFH sem re-shaping — chamado a cada frame durante animação.
    fn update_snfh_label_positions(&mut self, size: PhysicalSize<u32>) {
        if self.labels_snfh.is_empty() { return; }
        let ease = {
            let t = self.snfh_anim_t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let w    = size.width  as f32;
        let h    = size.height as f32;
        let y_ct = h * 0.14;
        let box_w = (w * 0.175).max(190.0).min(240.0);
        let box_h = 92.0_f32;
        let pad   = 12.0_f32;
        // Posição interpolada do canto do box SNFH
        let sx = (w - box_w - 24.0) + (24.0 - (w - box_w - 24.0)) * ease;
        let sy = y_ct + (box_h + 12.0) * ease;
        // 3 labels: título (idx 0), volume (idx 1), descritor (idx 2)
        let offsets: [(f32, f32); 3] = [(pad, 20.0), (pad, 38.0), (pad, 62.0)];
        for (lbl, (ox, oy)) in self.labels_snfh.iter_mut().zip(offsets.iter()) {
            lbl.x = sx + ox;
            lbl.y = sy + *oy;
        }
    }

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
        let mut snfh:   Vec<Label> = Vec::new();

        // ── Barra de menu — itens do topo ────────────────────────────────────
        {
            let menu_col = Color::rgb(196, 208, 224);
            let bar_y    = (MENU_BAR_H - 11.5_f32 * 1.25) / 2.0;
            let tops = [("Arquivo", MENU_TOP_XS[0]), ("Casos", MENU_TOP_XS[1]), ("Sobre", MENU_TOP_XS[2])];
            for (text, lx) in &tops {
                always.push(Label::new(fs, text, 11.5, menu_col, lx + 10.0, bar_y.max(4.0)));
            }
        }

        // ── Título ──────────────────────────────────────────────────────────
        let mut title = Label::new_bold(fs, "NeuroScan", 28.0, Self::col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        // Garante que o título fica abaixo da barra de menu
        title.y = (h * 0.04).max(MENU_BAR_H + 8.0);

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
        let _box_h = 92.0_f32;
        let pad    = 12.0_f32;

        // Cor da microlegenda — mais clara que col_dim para contraste adequado no fundo escuro
        let col_micro = Color::rgb(148, 164, 182);

        // ET — canto superior esquerdo
        {
            let ex = 24.0;
            let ey = y_ct;
            let vol = if self.scan.et_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.et_volume_ml)
            } else { String::new() };
            always.push(Label::new(fs,
                "\u{25CF}  ET  \u{00B7}  Enhancing Tumor",
                10.5, Self::rgb_f(ET_COLOR), ex + pad, ey + 20.0));
            if !vol.is_empty() {
                always.push(Label::new_bold(fs, &vol, 13.0, Color::WHITE, ex + pad, ey + 38.0));
            }
            always.push(Label::new(fs,
                "Realce pos-contraste · barreira comprometida",
                8.8, col_micro, ex + pad, ey + 62.0));
        }

        // SNFH — labels em vec separado; posição será definida por update_snfh_label_positions()
        {
            let vol_str = if self.scan.snfh_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.snfh_volume_ml)
            } else { String::new() };
            // Posição inicial qualquer — update_snfh_label_positions corrige antes do próximo frame
            snfh.push(Label::new(fs,
                "\u{25CF}  SNFH  \u{00B7}  Peritumoral Edema",
                10.5, Self::rgb_f(SNFH_COLOR), 0.0, 0.0));
            snfh.push(Label::new_bold(fs, &vol_str, 13.0, Color::WHITE, 0.0, 0.0));
            snfh.push(Label::new(fs,
                "Edema e infiltracao peritumoral",
                8.8, col_micro, 0.0, 0.0));
        }

        // NETC — centro inferior
        {
            let nx = (w / 2.0 - box_w / 2.0).max(24.0);
            let ny = (h - 185.0).max(y_ct + 92.0 + 20.0);
            let vol = if self.scan.netc_volume_ml > 0.0 {
                format!("{:.1} mL", self.scan.netc_volume_ml)
            } else { String::new() };
            always.push(Label::new(fs,
                "\u{25CF}  NETC  \u{00B7}  Necrotic Core",
                10.5, Self::rgb_f(NETC_COLOR), nx + pad, ny + 20.0));
            if !vol.is_empty() {
                always.push(Label::new_bold(fs, &vol, 13.0, Color::WHITE, nx + pad, ny + 38.0));
            }
            always.push(Label::new(fs,
                "Nucleo necrotico hipóxico · centro da lesao",
                8.8, col_micro, nx + pad, ny + 62.0));
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

        // ── Marca d'água "NeuroScan" — visível apenas na cena 3D, muito opaco ──
        {
            let mut wm = Label::new_bold(fs, "NeuroScan", 96.0,
                Color::rgba(210, 222, 238, 11), 0.0, 0.0);
            wm.x = (w - wm.measured_width()) / 2.0;
            wm.y = (h - wm.line_height()) / 2.0 + 20.0;
            always.push(wm);
        }

        // ── Rodapé técnico (canto inferior direito) ───────────────────────────
        {
            let footer = format!(
                "NeuroScan AI  \u{00B7}  v{}  \u{00B7}  nnUNet 2D  \u{00B7}  BraTS 2021  \u{00B7}  Dice 0.865",
                env!("CARGO_PKG_VERSION"));
            let mut ft = Label::new(fs, &footer, 9.0, Self::col_section(), 0.0, 0.0);
            ft.x = (w - ft.measured_width() - 18.0).max(0.0);
            ft.y = h - 16.0;
            always.push(ft);
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
        line_text!("WHO 2021  ·  Grau IV",   Self::col_value(),  11.0);
        line_text!("Glioblastoma Multiforme", Self::col_header(), 11.5);
        py += 6.0;

        section!("RISCO CLINICO");
        kv!("Grau WHO",      "IV  —  Alto Risco");
        kv!("IDH Status",    "Wild-type (wt)");
        kv!("MGMT Metil.",   "A investigar");
        kv!("Ki-67",         "> 30%");
        kv!("Critérios",     "RANO 2010");
        kv!("Sobrevida med.","14 – 16 meses");
        py += 3.0;
        panel.push(Label::new(fs,
            "* Dados pop. BraTS 2021. Nao substitui laudo.",
            8.5, Self::col_section(), pl, py));
        py += 14.0;
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
        kv!("Segmentacao", "nnUNet 2D Slice");
        kv!("Treinamento",  "484 casos BraTS 2021");
        kv!("Acuracia",     "Dice 0.865");
        kv!("Modalidades",  "FLAIR/T1w/T1ce/T2w");
        kv!("Resolucao",    "1mm isotropico");
        kv!("Referência",   "WHO CNS 2021");
        py += 4.0;
        panel.push(Label::new(fs,
            "Inferencia slice-a-slice com 4",
            9.5, Self::col_dim(), pl, py)); py += 13.0;
        panel.push(Label::new(fs,
            "canais MRI simultaneos (ONNX).",
            9.5, Self::col_dim(), pl, py)); py += 13.0;
        py += 6.0;
        panel.push(Label::new(fs,
            format!("NeuroScan AI  v{}", env!("CARGO_PKG_VERSION")).as_str(),
            9.0, Self::col_section(), pl, py));

        self.labels_always = always;
        self.labels_panel  = panel;
        self.labels_snfh   = snfh;
    }

    // -----------------------------------------------------------------------
    // Primitivas 2D
    // -----------------------------------------------------------------------

    fn build_primitives(
        &self,
        mvp:          &[[f32; 4]; 4],
        w:             f32,
        h:             f32,
        pulse_t:       f32,
        transition:    f32,
        spinner_angle: f32,
        infer_active:  bool,
    ) -> Prim2DBatch {
        let mut b    = Prim2DBatch::new();
        let y_ct     = h * 0.14;
        let box_w    = (w * 0.175).max(190.0).min(240.0);
        let box_h    = 92.0_f32;
        let bg       = [0.04, 0.06, 0.11, 0.92_f32];
        // Overlay interno sutil — frosted glass imperceptível que separa o texto do fundo
        let inner_ov = [0.20, 0.26, 0.40, 0.09_f32];
        // Pulso: raio do ponto de ancoragem oscila suavemente
        let dot_r    = 3.5 + 1.5 * (pulse_t * std::f32::consts::TAU * PULSE_FREQ).sin().abs();

        // ET callout (top-left)
        let et_x = 24.0;
        let et_y = y_ct;
        b.rect(et_x, et_y, box_w, box_h, bg, w, h);
        b.rect(et_x, et_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
        b.rect(et_x, et_y, box_w, 2.0, [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 1.0], w, h);
        if let Some((cx, cy)) = Self::project_to_screen(self.centroids[0], mvp, w, h) {
            b.line(et_x + box_w, et_y + box_h * 0.5, cx, cy,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.60], 1.5, w, h);
            b.rect(cx - dot_r, cy - dot_r, dot_r*2.0, dot_r*2.0,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.90], w, h);
        }

        // SNFH callout — posição interpolada suavemente (direita → esquerda quando painel abre)
        let ease_snfh = {
            let t = self.snfh_anim_t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let snfh_x = (w - box_w - 24.0) + (24.0 - (w - box_w - 24.0)) * ease_snfh;
        let snfh_y = y_ct + (y_ct + box_h + 12.0 - y_ct) * ease_snfh;
        // Ponto de saída da linha de ancoragem: direita do box quando à esquerda, esquerda quando à direita
        let snfh_line_x = snfh_x + box_w * ease_snfh;  // sai do lado direito ao animar para esquerda
        b.rect(snfh_x, snfh_y, box_w, box_h, bg, w, h);
        b.rect(snfh_x, snfh_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
        b.rect(snfh_x, snfh_y, box_w, 2.0, [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 1.0], w, h);
        if let Some((cx, cy)) = Self::project_to_screen(self.centroids[1], mvp, w, h) {
            b.line(snfh_line_x, snfh_y + box_h * 0.5, cx, cy,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.60], 1.5, w, h);
            b.rect(cx - dot_r, cy - dot_r, dot_r*2.0, dot_r*2.0,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.90], w, h);
        }

        // NETC callout (bottom-center)
        let netc_x = (w / 2.0 - box_w / 2.0).max(24.0);
        let netc_y = (h - 185.0).max(y_ct + box_h + 20.0);
        b.rect(netc_x, netc_y, box_w, box_h, bg, w, h);
        b.rect(netc_x, netc_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
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
        }

        // Overlay de inferência real — spinner grande centralizado
        if infer_active {
            b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, 0.72], w, h);
            let cx = w / 2.0;
            let cy = h / 2.0;
            let r  = 52.0_f32;
            // Trilha
            for i in 0..48usize {
                let a0 = (i     as f32 / 48.0) * std::f32::consts::TAU;
                let a1 = ((i+1) as f32 / 48.0) * std::f32::consts::TAU;
                b.line(cx + r * a0.cos(), cy + r * a0.sin(),
                       cx + r * a1.cos(), cy + r * a1.sin(),
                       [0.16, 0.24, 0.40, 0.28], 1.2, w, h);
            }
            // Arco com cauda
            let arc_span = std::f32::consts::TAU * (250.0 / 360.0);
            let tail_a   = spinner_angle - arc_span;
            for j in 0..36usize {
                let t0 = j     as f32 / 36.0;
                let t1 = (j+1) as f32 / 36.0;
                let a0 = tail_a + t0 * arc_span;
                let a1 = tail_a + t1 * arc_span;
                let br = t0.powf(1.3);
                b.line(cx + r * a0.cos(), cy + r * a0.sin(),
                       cx + r * a1.cos(), cy + r * a1.sin(),
                       [0.48 + 0.18 * br, 0.70 + 0.12 * br, 1.0, br * 0.90],
                       2.5, w, h);
            }
            let hx = cx + r * spinner_angle.cos();
            let hy = cy + r * spinner_angle.sin();
            let hr = 3.2_f32;
            b.rect(hx - hr, hy - hr, hr * 2.0, hr * 2.0, [0.72, 0.90, 1.0, 0.96], w, h);
        }

        // Overlay de transição — dissolve sinusoidal suave
        if transition > 0.0 {
            let alpha = (transition * std::f32::consts::PI).sin() * 0.92;
            b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, alpha], w, h);

            // Arc spinner — arco orbital girando no centro da tela
            // Visível apenas enquanto o overlay está suficientemente opaco (fase 0..0.9)
            let spinner_alpha = ((transition * std::f32::consts::PI).sin() * 1.6).min(1.0);
            if spinner_alpha > 0.05 {
                let cx = w / 2.0;
                let cy = h / 2.0;
                let r  = 36.0_f32;

                // Trilha circular fina
                let track_n = 48_usize;
                for i in 0..track_n {
                    let a0 = (i     as f32 / track_n as f32) * std::f32::consts::TAU;
                    let a1 = ((i+1) as f32 / track_n as f32) * std::f32::consts::TAU;
                    b.line(
                        cx + r * a0.cos(), cy + r * a0.sin(),
                        cx + r * a1.cos(), cy + r * a1.sin(),
                        [0.18, 0.26, 0.42, 0.28 * spinner_alpha],
                        1.0, w, h,
                    );
                }

                // Arco com cauda — 240° de varredura, cabeça em spinner_angle
                let arc_n    = 32_usize;
                let arc_span = std::f32::consts::TAU * (240.0 / 360.0);
                let tail_a   = spinner_angle - arc_span;
                for i in 0..arc_n {
                    let t0 = i     as f32 / arc_n as f32;
                    let t1 = (i+1) as f32 / arc_n as f32;
                    let a0 = tail_a + t0 * arc_span;
                    let a1 = tail_a + t1 * arc_span;
                    // Brilho cresce do tail (0) para a head (1) com curva suave
                    let brightness = t0.powf(1.4);
                    let seg_alpha  = brightness * 0.92 * spinner_alpha;
                    b.line(
                        cx + r * a0.cos(), cy + r * a0.sin(),
                        cx + r * a1.cos(), cy + r * a1.sin(),
                        [0.50 + 0.15 * brightness, 0.72 + 0.10 * brightness, 1.0, seg_alpha],
                        2.2, w, h,
                    );
                }

                // Ponto brilhante na cabeça do arco
                let hx  = cx + r * spinner_angle.cos();
                let hy  = cy + r * spinner_angle.sin();
                let hr  = 2.8 * spinner_alpha;
                b.rect(hx - hr, hy - hr, hr * 2.0, hr * 2.0,
                    [0.75, 0.92, 1.0, 0.95 * spinner_alpha], w, h);
            }
        }

        // ── Menu bar — renderizada por último: fica sobre tudo ────────────────
        // Fundo e separador inferior
        b.rect(0.0, 0.0, w, MENU_BAR_H, [0.04, 0.06, 0.12, 0.97], w, h);
        b.rect(0.0, MENU_BAR_H - 1.0, w, 1.0, [0.18, 0.26, 0.44, 0.45], w, h);

        // Highlight do item ativo (menu aberto) e do item hovered
        for i in 0..3_usize {
            let ix = MENU_TOP_XS[i];
            let iw = MENU_TOP_WS[i];
            let is_open    = self.menu_open == i as i32;
            let is_hovered = self.menu_hover_top == i as i32;
            if is_open {
                b.rect(ix, 0.0, iw, MENU_BAR_H, [0.20, 0.32, 0.55, 0.82], w, h);
            } else if is_hovered {
                b.rect(ix, 0.0, iw, MENU_BAR_H, [0.14, 0.22, 0.40, 0.55], w, h);
            }
        }

        // Dropdown do menu aberto
        if self.menu_open >= 0 {
            let mid      = self.menu_open as usize;
            let drop_x   = MENU_TOP_XS[mid];
            let drop_h   = self.dropdown_height(self.menu_open);
            let entries  = self.build_menu_entries(self.menu_open);

            // Fundo e bordas
            b.rect(drop_x, MENU_BAR_H, MENU_DROP_W, drop_h, [0.05, 0.08, 0.15, 0.97], w, h);
            b.rect(drop_x, MENU_BAR_H, MENU_DROP_W, 1.0,    [0.22, 0.32, 0.52, 0.55], w, h);
            b.rect(drop_x, MENU_BAR_H + drop_h, MENU_DROP_W, 1.0, [0.22, 0.32, 0.52, 0.55], w, h);
            b.rect(drop_x, MENU_BAR_H, 1.0, drop_h,              [0.22, 0.32, 0.52, 0.55], w, h);
            b.rect(drop_x + MENU_DROP_W - 1.0, MENU_BAR_H, 1.0, drop_h, [0.22, 0.32, 0.52, 0.55], w, h);

            // Items e separadores
            let mut iy = MENU_BAR_H;
            for (idx, (_, _, is_sep)) in entries.iter().enumerate() {
                if *is_sep {
                    let sep_y = iy + MENU_SEP_H / 2.0;
                    b.rect(drop_x + 8.0, sep_y, MENU_DROP_W - 16.0, 1.0, [0.18, 0.26, 0.42, 0.38], w, h);
                    iy += MENU_SEP_H;
                } else {
                    if self.menu_hover_item == idx as i32 {
                        b.rect(drop_x + 1.0, iy, MENU_DROP_W - 2.0, MENU_ITEM_H,
                            [0.16, 0.26, 0.46, 0.80], w, h);
                    }
                    iy += MENU_ITEM_H;
                }
            }
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
        // with_visible(false): evita o flash branco antes do primeiro frame wgpu
        let mut attrs = WindowAttributes::default()
            .with_title("NeuroScan — Visualizador Medico 3D")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_visible(false);
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

        // Construir labels da splash (rápido — sem IO)
        let size = window.inner_size();
        self.build_splash_labels(size);

        // Carregar todos os meshes em thread de fundo para não bloquear a splash
        let device = self.gpu.as_ref().unwrap().device.clone();
        let (tx, rx) = mpsc::channel::<Vec<LoadedMesh>>();
        std::thread::spawn(move || {
            // Tumores do caso inicial — slots 0..TUMOR_COUNT
            let case_dir = format!("{}/{}", CASES_DIR, TOP_CASES[0]);
            let tumor_defs = [
                (format!("{}/tumor_et.obj",   case_dir), ET_COLOR,   1.0_f32),
                (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0_f32),
                (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0_f32),
            ];
            let mut all: Vec<LoadedMesh> = tumor_defs.iter()
                .filter_map(|(path, tint, alpha)| {
                    Mesh::from_obj(&device, path).ok()
                        .map(|m| LoadedMesh { mesh: m, tint: *tint, alpha: *alpha })
                })
                .collect();
            // Cérebro permanente a seguir
            for (path, tint, alpha) in BRAIN_DEFS {
                if let Ok(m) = Mesh::from_obj(&device, path) {
                    all.push(LoadedMesh { mesh: m, tint: *tint, alpha: *alpha });
                }
            }
            let _ = tx.send(all);
        });
        self.splash_rx = Some(rx);
        info!("splash iniciada — carregamento de meshes em background");

        self.camera.target   = glam::Vec3::ZERO;
        self.camera.distance = 4.0;
        self.last_frame       = Instant::now();
        self.last_interaction = Instant::now();

        // Renderizar um frame escuro ANTES de mostrar a janela:
        // with_visible(false) + render síncrono + set_visible(true)
        // elimina o flash branco do OS antes do primeiro frame wgpu.
        // Nota: janelas invisíveis no Windows não recebem RedrawRequested,
        // por isso forçamos o primeiro render aqui, fora do loop de eventos.
        self.window = Some(Arc::clone(&window));
        {
            let cam = self.camera.build_uniform(size.width, size.height);
            let sw  = size.width  as f32;
            let sh  = size.height as f32;
            let mut first_prims = Prim2DBatch::new();
            // Fundo escuro idêntico ao da splash
            first_prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, 1.0], sw, sh);
            // Usar labels da splash já construídas acima
            let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
            if let Some(gpu) = &mut self.gpu {
                if let Err(e) = gpu.render(&cam, &[], &label_refs, &first_prims) {
                    warn!(error = %e, "erro no frame inicial da splash");
                }
            }
        }
        window.set_visible(true);
        self.window_shown = true;
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => { info!("close"); event_loop.exit(); }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    // Fechar menu ao pressionar qualquer tecla
                    if self.menu_open >= 0 {
                        self.menu_open       = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu     = Vec::new();
                    }
                    self.last_interaction = Instant::now();
                    match &event.logical_key {
                        Key::Character(ch) => match ch.as_str() {
                            "i" | "I" => {
                                self.show_panel = !self.show_panel;
                                // snfh_anim_t será avançado no loop de render para animar a transição
                            }
                            "o" | "O" => {
                                // Abrir file picker nativo em thread separada — não bloqueia o render loop
                                if self.dialog_rx.is_none() && !self.infer_active {
                                    let (tx, rx) = mpsc::channel();
                                    std::thread::spawn(move || {
                                        let path = rfd::FileDialog::new()
                                            .set_title("Selecionar volume NIfTI (.nii.gz)")
                                            .add_filter("NIfTI", &["gz", "nii"])
                                            .pick_file();
                                        if let Some(p) = path { let _ = tx.send(p); }
                                    });
                                    self.dialog_rx = Some(rx);
                                }
                            }
                            _ => {}
                        },
                        Key::Named(NamedKey::ArrowRight) => self.navigate_case( 1),
                        Key::Named(NamedKey::ArrowLeft)  => self.navigate_case(-1),
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Pressed {
                    let (mx, my) = self.mouse_pos.unwrap_or((0.0, 0.0));

                    // ── Click na barra de menu (top items) ────────────────────
                    let clicked_top = if my < MENU_BAR_H as f64 {
                        MENU_TOP_XS.iter().zip(MENU_TOP_WS.iter()).enumerate()
                            .find(|(_, (x, w))| mx >= **x as f64 && mx < (**x + **w) as f64)
                            .map_or(-1, |(i, _)| i as i32)
                    } else { -1 };

                    if clicked_top >= 0 {
                        self.menu_open = if self.menu_open == clicked_top { -1 } else { clicked_top };
                        self.menu_hover_item = -1;
                        let sz = self.window.as_ref().map(|w| w.inner_size())
                            .unwrap_or(PhysicalSize::new(1280, 720));
                        self.rebuild_menu_labels(sz);
                        if let Some(w) = &self.window { w.request_redraw(); }
                        return;
                    }

                    // ── Click num item do dropdown ────────────────────────────
                    if self.menu_open >= 0 && self.menu_hover_item >= 0 {
                        let menu_id  = self.menu_open;
                        let item_idx = self.menu_hover_item;
                        let entries  = self.build_menu_entries(menu_id);
                        self.menu_open       = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu     = Vec::new();
                        // Executar ação
                        if let Some((_, _, is_sep)) = entries.get(item_idx as usize) {
                            if !is_sep {
                                match (menu_id, item_idx) {
                                    (0, 0) => { // Arquivo → Abrir Volume NIfTI
                                        if self.dialog_rx.is_none() && !self.infer_active {
                                            let (tx, rx) = mpsc::channel();
                                            std::thread::spawn(move || {
                                                let path = rfd::FileDialog::new()
                                                    .set_title("Selecionar volume NIfTI (.nii.gz)")
                                                    .add_filter("NIfTI", &["gz", "nii"])
                                                    .pick_file();
                                                if let Some(p) = path { let _ = tx.send(p); }
                                            });
                                            self.dialog_rx = Some(rx);
                                        }
                                    }
                                    (0, 2) => { // Arquivo → Sair
                                        event_loop.exit();
                                        return;
                                    }
                                    (1, 0) => self.navigate_case(-1), // Casos → Anterior
                                    (1, 1) => self.navigate_case(1),  // Casos → Proximo
                                    (1, n) if n >= 3 => {             // Casos → item da lista
                                        let case_idx = (n - 3) as usize;
                                        if case_idx < TOP_CASES.len() && case_idx != self.current_case {
                                            self.transition_target = case_idx;
                                            self.transition_phase  = f32::EPSILON;
                                            self.spinner_angle     = 0.0;
                                            self.last_interaction  = Instant::now();
                                            let device  = self.gpu.as_ref().unwrap().device.clone();
                                            let case_id = TOP_CASES[case_idx].to_string();
                                            let (tx, rx) = mpsc::channel();
                                            std::thread::spawn(move || {
                                                let case_dir = format!("{}/{}", CASES_DIR, case_id);
                                                let defs = [
                                                    (format!("{}/tumor_et.obj",   case_dir), ET_COLOR,   1.0_f32),
                                                    (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0_f32),
                                                    (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0_f32),
                                                ];
                                                let meshes: Vec<LoadedMesh> = defs.iter()
                                                    .filter_map(|(path, tint, alpha)| {
                                                        Mesh::from_obj(&device, path).ok()
                                                            .map(|m| LoadedMesh { mesh: m, tint: *tint, alpha: *alpha })
                                                    })
                                                    .collect();
                                                let _ = tx.send(meshes);
                                            });
                                            self.loading_rx = Some(rx);
                                        }
                                    }
                                    _ => {} // Sobre: informacional, sem ação
                                }
                            }
                        }
                        if let Some(w) = &self.window { w.request_redraw(); }
                        return;
                    }

                    // ── Click fora do menu: fecha dropdown ────────────────────
                    if self.menu_open >= 0 {
                        self.menu_open       = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu     = Vec::new();
                        if let Some(w) = &self.window { w.request_redraw(); }
                    }
                }

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
                let mx = position.x;
                let my = position.y;

                if self.mouse_pressed {
                    self.last_interaction = Instant::now();
                    if let Some((px, py)) = self.mouse_pos {
                        self.camera.yaw   += (mx - px) as f32 * MOUSE_SENSITIVITY;
                        self.camera.pitch  = (self.camera.pitch - (my - py) as f32 * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((mx, my));

                // ── Hover tracking da menu bar ─────────────────────────────
                let new_top = if my < MENU_BAR_H as f64 {
                    MENU_TOP_XS.iter().zip(MENU_TOP_WS.iter()).enumerate()
                        .find(|(_, (x, w))| mx >= **x as f64 && mx < (**x + **w) as f64)
                        .map_or(-1, |(i, _)| i as i32)
                } else { -1 };

                let new_item = if self.menu_open >= 0 {
                    let mid    = self.menu_open as usize;
                    let drop_x = MENU_TOP_XS[mid] as f64;
                    let entries = self.build_menu_entries(self.menu_open);
                    let drop_h: f64 = entries.iter()
                        .map(|(_, _, s)| if *s { MENU_SEP_H } else { MENU_ITEM_H })
                        .sum::<f32>() as f64 + 4.0;
                    if mx >= drop_x && mx < drop_x + MENU_DROP_W as f64
                        && my >= MENU_BAR_H as f64 && my < MENU_BAR_H as f64 + drop_h
                    {
                        let rel = my - MENU_BAR_H as f64;
                        let mut y = 0.0_f64;
                        entries.iter().enumerate().find_map(|(i, (_, _, is_sep))| {
                            let ih = if *is_sep { MENU_SEP_H } else { MENU_ITEM_H } as f64;
                            if rel >= y && rel < y + ih { Some(i as i32) }
                            else { y += ih; None }
                        }).unwrap_or(-1)
                    } else { -1 }
                } else { -1 };

                if new_top != self.menu_hover_top || new_item != self.menu_hover_item {
                    self.menu_hover_top  = new_top;
                    self.menu_hover_item = new_item;
                    if let Some(w) = &self.window { w.request_redraw(); }
                }
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
                if self.splash_done {
                    self.build_labels(new_size);
                    self.update_snfh_label_positions(new_size);
                    self.rebuild_menu_labels(new_size);
                } else {
                    self.build_splash_labels(new_size);
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32().min(0.05);
                self.last_frame = Instant::now();

                // ─────────────────────────────────────────────────────────────
                // SPLASH SCREEN — exibida enquanto os meshes carregam em fundo
                // ─────────────────────────────────────────────────────────────
                if !self.splash_done {
                    self.splash_t      += dt;
                    self.spinner_angle += dt * std::f32::consts::TAU * 0.55;

                    // Checar chegada das meshes
                    if let Some(rx) = &self.splash_rx {
                        if let Ok(new_meshes) = rx.try_recv() {
                            self.meshes = new_meshes;
                            for i in 0..TUMOR_COUNT {
                                self.centroids[i] = self.meshes.get(i)
                                    .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                            }
                            self.scan = ScanMeta::load(
                                &ScanMeta::case_path(TOP_CASES[0]));
                            self.splash_rx = None;
                            // Construir labels do viewer agora para o fade-out
                            let sz = PhysicalSize::new(
                                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                                self.gpu.as_ref().map_or(720,  |g| g.config.height),
                            );
                            self.build_labels(sz);
                            self.update_snfh_label_positions(sz);
                            self.rebuild_menu_labels(sz);
                            info!(meshes = self.meshes.len(), "splash: meshes prontas");
                        }
                    }

                    // Avançar fade-out apenas quando meshes chegaram
                    if self.splash_rx.is_none() {
                        self.splash_fade += dt / SPLASH_FADEOUT_DURATION;
                        if self.splash_fade >= 1.0 {
                            self.splash_done  = true;
                            self.splash_fade  = 1.0;
                            self.spinner_angle = 0.0;
                        }
                    }

                    let sz = PhysicalSize::new(
                        self.gpu.as_ref().map_or(1280, |g| g.config.width),
                        self.gpu.as_ref().map_or(720,  |g| g.config.height),
                    );
                    let sw = sz.width  as f32;
                    let sh = sz.height as f32;
                    let cam = self.camera.build_uniform(sz.width, sz.height);

                    if self.splash_rx.is_some() {
                        // Splash puro: sem 3D, apenas animação + título
                        let prims      = self.build_splash_primitives(sw, sh);
                        let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
                        if let Some(gpu) = &mut self.gpu {
                            if let Err(e) = gpu.render(&cam, &[], &label_refs, &prims) {
                                warn!(error = %e, "erro render splash");
                            }
                        }
                        // Revelar a janela após o primeiro frame renderizado — sem flash branco
                        if !self.window_shown {
                            if let Some(w) = &self.window { w.set_visible(true); }
                            self.window_shown = true;
                        }
                    } else {
                        // Fade-out: cena 3D completa revelando por baixo do overlay
                        let overlay = (1.0 - self.splash_fade).max(0.0);
                        let pulse_t  = self.pulse_t;
                        let spinner  = self.spinner_angle;
                        let mut prims = self.build_primitives(
                            &cam.mvp, sw, sh, pulse_t, 0.0, spinner, false);
                        if overlay > 0.01 {
                            prims.rect(0.0, 0.0, sw, sh,
                                [0.03, 0.04, 0.08, overlay], sw, sh);
                        }
                        let label_refs: Vec<&Label> = self.labels_always.iter().collect();
                        if let Some(gpu) = &mut self.gpu {
                            let entries: Vec<MeshEntry> = self.meshes.iter()
                                .map(|m| MeshEntry { mesh: &m.mesh, tint: m.tint, alpha: m.alpha })
                                .collect();
                            if let Err(e) = gpu.render(&cam, &entries, &label_refs, &prims) {
                                warn!(error = %e, "erro render fade-out splash");
                            }
                        }
                    }

                    if let Some(w) = &self.window { w.request_redraw(); }
                    return;
                }

                // ── File dialog: checar se usuário selecionou arquivo ──────
                let picked_path: Option<std::path::PathBuf> = if let Some(rx) = &self.dialog_rx {
                    match rx.try_recv() {
                        Ok(p)  => { self.dialog_rx = None; Some(p) }
                        Err(_) => None,
                    }
                } else { None };

                if let Some(path) = picked_path {
                    // Disparar inferência Python em background
                    let path_str   = path.display().to_string();
                    let out_dir    = "assets/models/infer".to_string();
                    let device_bg  = self.gpu.as_ref().unwrap().device.clone();
                    let (tx, rx)   = mpsc::channel::<bool>();
                    let out_clone  = out_dir.clone();
                    std::thread::spawn(move || {
                        info!(input = %path_str, "iniciando inferencia Python");
                        let status = std::process::Command::new("python")
                            .args(["scripts/ml/infer_single.py",
                                   "--input",      &path_str,
                                   "--output-dir", &out_clone,
                                   "--model",      "assets/models/onnx/nnunet_brats_4ch.onnx",
                                   "--meta",       "assets/models/brain_meta.json"])
                            .status();
                        let ok = matches!(status, Ok(s) if s.success());
                        // Quando Python terminar, carregar os OBJs em GPU na mesma thread
                        if ok {
                            // Pré-carregar meshes para enviar já prontas
                            let defs = [
                                (format!("{}/tumor_et.obj",   out_clone), ET_COLOR,   1.0_f32),
                                (format!("{}/tumor_snfh.obj", out_clone), SNFH_COLOR, 1.0_f32),
                                (format!("{}/tumor_netc.obj", out_clone), NETC_COLOR, 1.0_f32),
                            ];
                            let _meshes: Vec<LoadedMesh> = defs.iter()
                                .filter_map(|(p, t, a)| {
                                    Mesh::from_obj(&device_bg, p).ok()
                                        .map(|m| LoadedMesh { mesh: m, tint: *t, alpha: *a })
                                })
                                .collect();
                            // Nota: enviar meshes via channel não é possível sem Arc<Mutex>
                            // (Device é Clone mas Buffer não é Send em todas as versões).
                            // Sinalizamos ok=true e o main thread carrega do disco.
                        }
                        let _ = tx.send(ok);
                    });
                    self.infer_rx    = Some(rx);
                    self.infer_active = true;
                    info!("inferencia iniciada");
                }

                // ── Inferência: checar conclusão ────────────────────────────
                if let Some(rx) = &self.infer_rx {
                    if let Ok(ok) = rx.try_recv() {
                        self.infer_rx     = None;
                        self.infer_active = false;
                        if ok {
                            // Carregar meshes inferidas e substituir slots 0..2
                            let device = self.gpu.as_ref().unwrap().device.clone();
                            let out_dir = "assets/models/infer";
                            let defs = [
                                (format!("{}/tumor_et.obj",   out_dir), ET_COLOR,   1.0_f32),
                                (format!("{}/tumor_snfh.obj", out_dir), SNFH_COLOR, 1.0_f32),
                                (format!("{}/tumor_netc.obj", out_dir), NETC_COLOR, 1.0_f32),
                            ];
                            let new_meshes: Vec<LoadedMesh> = {
                                defs.iter()
                                    .filter_map(|(p, t, a)| {
                                        Mesh::from_obj(&device, p).ok()
                                            .map(|m| LoadedMesh { mesh: m, tint: *t, alpha: *a })
                                    })
                                    .collect()
                            };
                            for (i, lm) in new_meshes.into_iter().enumerate() {
                                if i < self.meshes.len() { self.meshes[i] = lm; }
                            }
                            for i in 0..TUMOR_COUNT {
                                self.centroids[i] = self.meshes.get(i)
                                    .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                            }
                            let infer_meta = format!("{}/scan_meta.json", out_dir);
                            self.scan = ScanMeta::load(&infer_meta);
                            let sz = PhysicalSize::new(
                                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                                self.gpu.as_ref().map_or(720,  |g| g.config.height),
                            );
                            self.build_labels(sz);
                            self.update_snfh_label_positions(sz);
                            self.rebuild_menu_labels(sz);
                            info!("caso inferido carregado com sucesso");
                        } else {
                            warn!("inferencia Python falhou ou foi cancelada");
                        }
                    }
                }

                // ── Spinner de inferência (independente da transição de caso) ──
                if self.infer_active {
                    self.spinner_angle += dt * std::f32::consts::TAU * 0.75;
                }

                // ── Microanimação: rotação automática após inatividade ──────
                if self.last_interaction.elapsed().as_secs_f32() > AUTO_ROTATE_IDLE_S {
                    self.camera.yaw += AUTO_ROTATE_SPEED * dt * 60.0;
                }

                // ── Pulso dos pontos de ancoragem ──────────────────────────
                self.pulse_t += dt;

                // ── Spinner: gira apenas enquanto há transição ativa ────────
                if self.transition_phase > 0.0 {
                    self.spinner_angle += dt * std::f32::consts::TAU * 0.75;
                }

                // ── Checar se a thread de carregamento terminou ─────────────
                let mut rebuild_needed = false;
                let meshes_arrived = if let Some(rx) = &self.loading_rx {
                    match rx.try_recv() {
                        Ok(new_meshes) => {
                            for (i, lm) in new_meshes.into_iter().enumerate() {
                                if i < self.meshes.len() { self.meshes[i] = lm; }
                            }
                            for i in 0..TUMOR_COUNT {
                                self.centroids[i] = self.meshes.get(i)
                                    .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                            }
                            self.current_case = self.transition_target;
                            self.scan = ScanMeta::load(
                                &ScanMeta::case_path(TOP_CASES[self.current_case]));
                            rebuild_needed = true;
                            info!(case = TOP_CASES[self.current_case], "caso carregado (bg)");
                            true
                        }
                        Err(_) => false,
                    }
                } else { false };
                if meshes_arrived { self.loading_rx = None; }

                // ── Avanço da fase de transição ─────────────────────────────
                // Fade-in (0→0.5): avança sempre.
                // Mantém ≤ 0.5 enquanto meshes não chegaram.
                // Fade-out (0.5→1.0): avança apenas após meshes prontas.
                if self.transition_phase > 0.0 {
                    let loading_done = self.loading_rx.is_none();
                    if self.transition_phase < 0.5 || loading_done {
                        self.transition_phase += dt / TRANSITION_DURATION;
                    }
                    self.transition_phase = self.transition_phase.min(if loading_done { 2.0 } else { 0.499 });
                    if self.transition_phase >= 1.0 {
                        self.transition_phase = 0.0;
                        self.spinner_angle    = 0.0;
                    }
                }

                let size = PhysicalSize::new(
                    self.gpu.as_ref().map_or(1280, |g| g.config.width),
                    self.gpu.as_ref().map_or(720,  |g| g.config.height),
                );

                // ── Animação do SNFH callout — apenas muta .x/.y nos labels existentes ──
                // Sem rebuild de font/shaping: zero custo por frame, animação perfeitamente fluida.
                let snfh_target = if self.show_panel { 1.0_f32 } else { 0.0_f32 };
                let snfh_prev   = self.snfh_anim_t;
                let snfh_speed  = dt / 0.28;
                self.snfh_anim_t = if self.snfh_anim_t < snfh_target {
                    (self.snfh_anim_t + snfh_speed).min(snfh_target)
                } else {
                    (self.snfh_anim_t - snfh_speed).max(snfh_target)
                };
                if (self.snfh_anim_t - snfh_prev).abs() > 0.001 {
                    self.update_snfh_label_positions(size);
                }

                if rebuild_needed {
                    self.build_labels(size);
                    self.update_snfh_label_positions(size);
                    self.rebuild_menu_labels(size);
                }

                let cam   = self.camera.build_uniform(size.width, size.height);
                let infer = self.infer_active;
                let prims = self.build_primitives(&cam.mvp, size.width as f32, size.height as f32,
                    self.pulse_t, self.transition_phase, self.spinner_angle, infer);

                let mut label_refs: Vec<&Label> = self.labels_always.iter().collect();
                label_refs.extend(self.labels_snfh.iter());
                if self.show_panel   { label_refs.extend(self.labels_panel.iter()); }
                if self.menu_open >= 0 { label_refs.extend(self.labels_menu.iter()); }

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
