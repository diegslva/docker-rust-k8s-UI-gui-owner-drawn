mod camera;
mod mesh;
mod renderer;
mod ui;

use anyhow::{Context, Result};
use camera::OrbitalCamera;
use mesh::Mesh;
use renderer::{GpuState, MeshEntry};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use ui::{Color, Label};

/// Metadados do scan lidos de assets/models/scan_meta.json
#[derive(Default)]
struct ScanMeta {
    case_id:         String,
    dataset:         String,
    modalities:      String,
    model_name:      String,
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
            model_name:      v["model_name"].as_str().unwrap_or("").to_string(),
            et_volume_ml:    v["et_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            snfh_volume_ml:  v["snfh_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            netc_volume_ml:  v["netc_volume_ml"].as_f64().unwrap_or(0.0) as f32,
            total_volume_ml: v["total_volume_ml"].as_f64().unwrap_or(0.0) as f32,
        }
    }
}
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorIcon, Icon, Window, WindowAttributes, WindowId};

const WINDOW_WIDTH:  f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8]  = include_bytes!("../assets/icon_256x256.png");

// --- Meshes ---
// --- Tumores (NIfTI, espaco normalizado) ---
const TUMOR_ET_OBJ:   &str = "assets/models/tumor_et.obj";
const TUMOR_SNFH_OBJ: &str = "assets/models/tumor_snfh.obj";
const TUMOR_NETC_OBJ: &str = "assets/models/tumor_netc.obj";

// --- Cerebro premium TurboSquid (16 partes anatomicas) ---
const PREMIUM_DIR: &str = "assets/models/premium";

// --- Cores RGB linear ---
// Cerebro: cor anatomica realista (rosa-acinzentado como tecido cerebral real)
const LH_COLOR:    [f32; 3] = [0.90, 0.72, 0.70]; // hemisferio esquerdo
const RH_COLOR:    [f32; 3] = [0.88, 0.70, 0.68]; // hemisferio direito (ligeiramente diferente)
const CEREB_COLOR: [f32; 3] = [0.82, 0.62, 0.60]; // cerebelo — mais escuro
const INNER_COLOR: [f32; 3] = [0.78, 0.65, 0.72]; // estruturas internas (talamo, etc.)
const VENTR_COLOR: [f32; 3] = [0.55, 0.72, 0.85]; // ventriculos — azul-aqua translucido
const ET_COLOR:    [f32; 3] = [0.95, 0.18, 0.12]; // vermelho vivo
const SNFH_COLOR:  [f32; 3] = [0.95, 0.78, 0.05]; // ambar dourado
const NETC_COLOR:  [f32; 3] = [0.25, 0.50, 0.98]; // azul eletrico

/// Alphas — hemisferios semi-transparentes, estruturas internas mais opacas
const LH_ALPHA:    f32 = 0.30;
const RH_ALPHA:    f32 = 0.30;
const CEREB_ALPHA: f32 = 0.55;
const INNER_ALPHA: f32 = 0.80;
const VENTR_ALPHA: f32 = 0.45;

// --- Camera ---
const MOUSE_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY:  f32 = 0.3;
const ZOOM_MIN:          f32 = 1.5;
const ZOOM_MAX:          f32 = 20.0;
const PITCH_LIMIT:       f32 = 1.5;

/// Definicao de cada mesh a carregar: (path, tint_rgb, alpha)
/// Ordem: opacos primeiro (tumores), depois alpha (cerebro de fora para dentro)
const MESH_DEFS: &[(&str, [f32; 3], f32)] = &[
    // Tumores — opacos, renderizados primeiro no depth buffer
    (TUMOR_ET_OBJ,   ET_COLOR,    1.0),
    (TUMOR_SNFH_OBJ, SNFH_COLOR,  1.0),
    (TUMOR_NETC_OBJ, NETC_COLOR,  1.0),
    // Estruturas internas — semi-opacas
    (concat!("assets/models/premium/", "Ventricles.obj"),                          VENTR_COLOR, VENTR_ALPHA),
    (concat!("assets/models/premium/", "Thalmus_and_Optic_Tract.obj"),             INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Corpus_Callosum.obj"),                     INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Hippocampus_and_Indusium_Griseum.obj"),    INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Putamen_and_Amygdala.obj"),                INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Globus_Pallidus_Externus.obj"),            INNER_COLOR, INNER_ALPHA),
    (concat!("assets/models/premium/", "Globus_Pallidus_Internus.obj"),            INNER_COLOR, INNER_ALPHA),
    // Cerebelo e tronco — semi-transparentes
    (concat!("assets/models/premium/", "Cerebellum.obj"),                          CEREB_COLOR, CEREB_ALPHA),
    (concat!("assets/models/premium/", "Medulla_and_Pons.obj"),                    CEREB_COLOR, CEREB_ALPHA),
    // Hemisferios — camada mais externa, mais transparentes
    (concat!("assets/models/premium/", "Left_Cerebral_Hemisphere.obj"),            LH_COLOR,    LH_ALPHA),
    (concat!("assets/models/premium/", "Right_Cerebral_Hemisphere.obj"),           RH_COLOR,    RH_ALPHA),
];

/// Legenda: (texto, cor) — tumores sao os 3 primeiros MESH_DEFS.
const LEGEND: &[(&str, [f32; 3])] = &[
    ("ET   Enhancing Tumor",     ET_COLOR),
    ("SNFH   Peritumoral Edema", SNFH_COLOR),
    ("NETC   Necrotic Core",     NETC_COLOR),
];
/// Quantos dos primeiros MESH_DEFS sao tumores (para a legenda)
const TUMOR_COUNT: usize = 3;

struct LoadedMesh {
    mesh:  Mesh,
    tint:  [f32; 3],
    alpha: f32,
}

struct App {
    window:        Option<Arc<Window>>,
    gpu:           Option<GpuState>,
    meshes:        Vec<LoadedMesh>,
    camera:        OrbitalCamera,
    labels:        Vec<Label>,
    scan:          ScanMeta,
    last_frame:    Instant,
    mouse_pressed: bool,
    mouse_pos:     Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, gpu: None, meshes: Vec::new(),
            camera: OrbitalCamera::new(4.0),
            labels: Vec::new(),
            scan: ScanMeta::load("assets/models/scan_meta.json"),
            last_frame: Instant::now(),
            mouse_pressed: false, mouse_pos: None,
        }
    }

    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w  = size.width  as f32;
        let h  = size.height as f32;
        let fs = gpu.font_system_mut();

        // Paleta de cores da UI
        let col_dim   = Color::rgb(120, 130, 148);
        let col_value = Color::rgb(220, 228, 240);

        // ── Titulo ────────────────────────────────────────────────────────
        let mut title = Label::new_bold(fs,
            "NeuroScan  \u{00B7}  Cerebro 3D + Tumor WHO 2021",
            30.0, Color::WHITE, 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.05;

        // Subtitulo: caso + modalidades reais
        let sub_text = if self.scan.case_id.is_empty() {
            "Arraste para girar  \u{00B7}  Scroll para zoom".to_string()
        } else {
            format!("{}  \u{00B7}  {}  \u{00B7}  {}  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom",
                self.scan.case_id, self.scan.dataset, self.scan.modalities)
        };
        let mut sub = Label::new(fs, &sub_text, 14.0, col_dim, 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 4.0;

        let mut labels = vec![title, sub];

        // ── Legenda canto inferior esquerdo com volumes reais ─────────────
        let legend_font = 13.5;
        let legend_gap  = legend_font * 1.6;
        let volumes = [
            self.scan.et_volume_ml,
            self.scan.snfh_volume_ml,
            self.scan.netc_volume_ml,
        ];
        let n_visible: f32 = LEGEND.iter().enumerate()
            .filter(|(i, _)| self.meshes.len() > *i)
            .count() as f32;
        let legend_start_y = h * 0.96 - n_visible * legend_gap;
        let mut row = 0;
        for (i, (text, rgb)) in LEGEND.iter().enumerate() {
            if self.meshes.len() <= i { continue; }
            let vol_str = if volumes[i] > 0.0 {
                format!("{}   {:.1} mL", text, volumes[i])
            } else {
                text.to_string()
            };
            labels.push(Label::new(fs, &vol_str, legend_font,
                Color::rgb((rgb[0]*255.0) as u8, (rgb[1]*255.0) as u8, (rgb[2]*255.0) as u8),
                28.0, legend_start_y + row as f32 * legend_gap));
            row += 1;
        }

        // ── Painel inferior direito: volume total + modelo ─────────────────
        let panel_x = w - 280.0;
        let panel_font = 12.5;
        let panel_gap  = panel_font * 1.7;
        let panel_items: &[(&str, &str)] = &[
            ("VOLUME TOTAL",  &format!("{:.1} mL", self.scan.total_volume_ml)),
            ("MODELO",        &self.scan.model_name.clone()),
            ("CLASSIFICACAO", "WHO 2021 · GBM"),
            ("DATASET",       "BraTS 2021 · 484 casos"),
        ];
        let panel_start_y = h * 0.96 - (panel_items.len() as f32) * panel_gap;
        for (j, (key, val)) in panel_items.iter().enumerate() {
            let y = panel_start_y + j as f32 * panel_gap;
            labels.push(Label::new(fs, key,  panel_font - 1.5, col_dim,    panel_x,        y));
            labels.push(Label::new(fs, val,  panel_font,       col_value,  panel_x,        y + panel_font * 0.9));
        }

        self.labels = labels;
    }

    fn reposition_labels(&mut self, size: PhysicalSize<u32>) {
        let w  = size.width  as f32;
        let h  = size.height as f32;
        let ty = h * 0.05;
        let lh = self.labels.first().map_or(30.0 * 1.25, |t| t.line_height());
        if let Some(t) = self.labels.get_mut(0) { t.x = (w - t.measured_width()) / 2.0; t.y = ty; }
        if let Some(s) = self.labels.get_mut(1) { s.x = (w - s.measured_width()) / 2.0; s.y = ty + lh + 4.0; }
        let legend_font = 14.0;
        let legend_gap  = legend_font * 1.45;
        let n = (self.labels.len().saturating_sub(2)) as f32;
        let ly = h * 0.97 - n * legend_gap;
        for (j, label) in self.labels.iter_mut().skip(2).enumerate() {
            label.y = ly + j as f32 * legend_gap;
        }
    }
}

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
            .with_title("NeuroScan [Phase 6 — Brain 3D Transparente + Tumor Multi-Classe]")
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

        let device = &self.gpu.as_ref().unwrap().device;
        for (path, tint, alpha) in MESH_DEFS {
            match Mesh::from_obj(device, path) {
                Ok(m)  => { info!(path, alpha, "mesh carregada"); self.meshes.push(LoadedMesh { mesh: m, tint: *tint, alpha: *alpha }); }
                Err(e) => warn!(error = %e, path, "mesh nao encontrada"),
            }
        }
        info!(total = self.meshes.len(), "scene carregada");

        self.camera.target   = glam::Vec3::ZERO;
        self.camera.distance = 4.0;

        let size = window.inner_size();
        self.build_labels(size);
        self.last_frame = Instant::now();
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => { info!("close"); event_loop.exit(); }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if let Some(w) = &self.window {
                        w.set_cursor(if self.mouse_pressed { CursorIcon::Grabbing } else { CursorIcon::Grab });
                    }
                    if state == ElementState::Released { self.mouse_pos = None; }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((px, py)) = self.mouse_pos {
                        self.camera.yaw   += (position.x - px) as f32 * MOUSE_SENSITIVITY;
                        self.camera.pitch  = (self.camera.pitch - (position.y - py) as f32 * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((position.x, position.y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.01,
                };
                self.camera.distance = (self.camera.distance - scroll * ZOOM_SENSITIVITY).clamp(ZOOM_MIN, ZOOM_MAX);
            }

            WindowEvent::Resized(new_size) => {
                debug!(w = new_size.width, h = new_size.height, "resize");
                if let Some(gpu) = &mut self.gpu { gpu.resize(new_size); }
                self.reposition_labels(new_size);
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::RedrawRequested => {
                self.last_frame = Instant::now();
                let w   = self.gpu.as_ref().map_or(1280, |g| g.config.width);
                let h   = self.gpu.as_ref().map_or(720,  |g| g.config.height);
                let cam = self.camera.build_uniform(w, h);

                if let Some(gpu) = &mut self.gpu {
                    let entries: Vec<MeshEntry> = self.meshes.iter()
                        .map(|m| MeshEntry { mesh: &m.mesh, tint: m.tint, alpha: m.alpha })
                        .collect();
                    let label_refs: Vec<&Label> = self.labels.iter().collect();
                    if let Err(e) = gpu.render(&cam, &entries, &label_refs) {
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
