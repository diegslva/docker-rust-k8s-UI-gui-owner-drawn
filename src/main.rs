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
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorIcon, Icon, Window, WindowAttributes, WindowId};

const WINDOW_WIDTH:  f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8]  = include_bytes!("../assets/icon_256x256.png");

// --- Meshes ---
const BRAIN_OBJ:      &str = "assets/models/brain.obj";
const TUMOR_ET_OBJ:   &str = "assets/models/tumor_et.obj";
const TUMOR_SNFH_OBJ: &str = "assets/models/tumor_snfh.obj";
const TUMOR_NETC_OBJ: &str = "assets/models/tumor_netc.obj";

// --- Cores RGB linear ---
const BRAIN_COLOR: [f32; 3] = [0.85, 0.75, 0.73]; // cinza rosado, substancia cinzenta
const ET_COLOR:    [f32; 3] = [0.95, 0.18, 0.12]; // vermelho vivo
const SNFH_COLOR:  [f32; 3] = [0.95, 0.78, 0.05]; // ambar dourado
const NETC_COLOR:  [f32; 3] = [0.25, 0.50, 0.98]; // azul eletrico

/// Alpha do cerebro — semi-transparente para o tumor aparecer por dentro.
const BRAIN_ALPHA: f32 = 0.35;

// --- Camera ---
const MOUSE_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY:  f32 = 0.3;
const ZOOM_MIN:          f32 = 1.5;
const ZOOM_MAX:          f32 = 20.0;
const PITCH_LIMIT:       f32 = 1.5;

/// Definicao de cada mesh a carregar: (path, tint_rgb, alpha)
const MESH_DEFS: &[(&str, [f32; 3], f32)] = &[
    (BRAIN_OBJ,      BRAIN_COLOR, BRAIN_ALPHA), // semi-transparente
    (TUMOR_ET_OBJ,   ET_COLOR,    1.0),
    (TUMOR_SNFH_OBJ, SNFH_COLOR,  1.0),
    (TUMOR_NETC_OBJ, NETC_COLOR,  1.0),
];

/// Legenda: (texto, cor) — so mostra se o mesh correspondente foi carregado.
/// Indice i aqui corresponde ao MESH_DEFS[i+1] (pulamos o cerebro).
const LEGEND: &[(&str, [f32; 3])] = &[
    ("ET   Enhancing Tumor",     ET_COLOR),
    ("SNFH   Peritumoral Edema", SNFH_COLOR),
    ("NETC   Necrotic Core",     NETC_COLOR),
];

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
            last_frame: Instant::now(),
            mouse_pressed: false, mouse_pos: None,
        }
    }

    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w  = size.width  as f32;
        let h  = size.height as f32;
        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(fs,
            "NeuroScan  \u{00B7}  Cerebro 3D + Tumor WHO 2021",
            30.0, Color::WHITE, 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.05;

        let mut sub = Label::new(fs,
            "Predicao nnUNet  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom",
            15.0, Color::rgb(140, 150, 165), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 4.0;

        let mut labels = vec![title, sub];

        // Legenda canto inferior esquerdo
        let legend_font = 14.0;
        let legend_gap  = legend_font * 1.45;
        let n_visible: f32 = LEGEND.iter().enumerate()
            .filter(|(i, _)| self.meshes.len() > i + 1)
            .count() as f32;
        let legend_start_y = h * 0.97 - n_visible * legend_gap;
        let mut row = 0;
        for (i, (text, rgb)) in LEGEND.iter().enumerate() {
            if self.meshes.len() <= i + 1 { continue; }
            labels.push(Label::new(fs, text, legend_font,
                Color::rgb((rgb[0]*255.0) as u8, (rgb[1]*255.0) as u8, (rgb[2]*255.0) as u8),
                28.0, legend_start_y + row as f32 * legend_gap));
            row += 1;
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
