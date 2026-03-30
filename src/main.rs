mod camera;
mod mesh;
mod renderer;
mod ui;

use anyhow::{Context, Result};
use camera::OrbitalCamera;
use mesh::Mesh;
use renderer::GpuState;
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

const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8] = include_bytes!("../assets/icon_256x256.png");

// --- Meshes NeuroScan ---
const BRAIN_OBJ: &str = "assets/models/brain.obj";
/// ET: tumor realçado (Enhancing Tumor)
const TUMOR_ET_OBJ: &str = "assets/models/tumor_et.obj";
/// SNFH: edema peritumoral (Surrounding Non-enhancing FLAIR Hyperintensity)
const TUMOR_SNFH_OBJ: &str = "assets/models/tumor_snfh.obj";
/// NETC: núcleo necrótico (Non-Enhancing Tumor Core)
const TUMOR_NETC_OBJ: &str = "assets/models/tumor_netc.obj";

// --- Cores RGB (linear) por sub-região tumoral (WHO 2021) ---
/// Cerebro: substancia cinzenta, cinza rosado
const BRAIN_COLOR: [f32; 3] = [0.82, 0.72, 0.70];
/// ET – vermelho vivo
const ET_COLOR: [f32; 3] = [0.95, 0.18, 0.12];
/// SNFH – âmbar/amarelo
const SNFH_COLOR: [f32; 3] = [0.95, 0.75, 0.08];
/// NETC – azul
const NETC_COLOR: [f32; 3] = [0.25, 0.45, 0.95];

/// Sensibilidade do drag de mouse (radianos por pixel).
const MOUSE_SENSITIVITY: f32 = 0.005;
/// Sensibilidade do zoom via scroll.
const ZOOM_SENSITIVITY: f32 = 0.3;
/// Limites de distancia da camera.
const ZOOM_MIN: f32 = 1.5;
const ZOOM_MAX: f32 = 20.0;
/// Limite vertical da camera (evita gimbal lock no polo).
const PITCH_LIMIT: f32 = 1.5;

/// Legenda: (texto, [f32; 3] cor do mesh, mesh_index_minimo para mostrar)
const LEGEND: &[(&str, [f32; 3])] = &[
    ("ET  Enhancing Tumor",     ET_COLOR),
    ("SNFH  Peritumoral Edema", SNFH_COLOR),
    ("NETC  Necrotic Core",     NETC_COLOR),
];

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    /// (mesh, tint_rgb) — indice 0=cerebro, 1=ET, 2=SNFH, 3=NETC
    meshes: Vec<(Mesh, [f32; 3])>,
    camera: OrbitalCamera,
    labels: Vec<Label>,
    last_frame: Instant,
    mouse_pressed: bool,
    mouse_pos: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            meshes: Vec::new(),
            camera: OrbitalCamera::new(4.0),
            labels: Vec::new(),
            last_frame: Instant::now(),
            mouse_pressed: false,
            mouse_pos: None,
        }
    }

    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let fs = gpu.font_system_mut();

        // --- Titulo ---
        let mut title = Label::new_bold(
            fs,
            "NeuroScan  \u{00B7}  Cerebro 3D + Tumor WHO 2021",
            30.0,
            Color::WHITE,
            0.0, 0.0,
        );
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.05;

        // --- Subtitulo ---
        let mut sub = Label::new(
            fs,
            "Predicao nnUNet  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom",
            15.0,
            Color::rgb(140, 150, 165),
            0.0, 0.0,
        );
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 4.0;

        let mut labels = vec![title, sub];

        // --- Legenda (canto inferior esquerdo) ---
        // Mostra apenas as classes que foram carregadas (indices 1, 2, 3).
        let legend_x = 28.0;
        let legend_font = 14.0;
        let legend_gap  = legend_font * 1.4;
        // Ancora no bottom: empilha de baixo para cima
        let legend_bottom = h * 0.97;
        let n_legend = LEGEND.len() as f32;
        let legend_start_y = legend_bottom - n_legend * legend_gap;

        for (i, (text, rgb)) in LEGEND.iter().enumerate() {
            // +1 porque indice 0 e o cerebro
            let mesh_idx = i + 1;
            if self.meshes.len() <= mesh_idx {
                continue; // essa classe nao foi carregada
            }
            let color = Color::rgb(
                (rgb[0] * 255.0) as u8,
                (rgb[1] * 255.0) as u8,
                (rgb[2] * 255.0) as u8,
            );
            labels.push(Label::new(
                fs,
                text,
                legend_font,
                color,
                legend_x,
                legend_start_y + i as f32 * legend_gap,
            ));
        }

        self.labels = labels;
    }

    fn reposition_labels(&mut self, size: PhysicalSize<u32>) {
        let w = size.width as f32;
        let h = size.height as f32;
        let title_y  = h * 0.05;
        let title_lh = self.labels.first().map_or(30.0 * 1.25, |t| t.line_height());

        // Titulo (idx 0)
        if let Some(t) = self.labels.get_mut(0) {
            t.x = (w - t.measured_width()) / 2.0;
            t.y = title_y;
        }
        // Subtitulo (idx 1)
        if let Some(s) = self.labels.get_mut(1) {
            s.x = (w - s.measured_width()) / 2.0;
            s.y = title_y + title_lh + 4.0;
        }
        // Legenda (idx 2+)
        let legend_font = 14.0;
        let legend_gap  = legend_font * 1.4;
        let legend_bottom = h * 0.97;
        let n_legend = (self.labels.len().saturating_sub(2)) as f32;
        let legend_start_y = legend_bottom - n_legend * legend_gap;
        for (j, label) in self.labels.iter_mut().skip(2).enumerate() {
            label.y = legend_start_y + j as f32 * legend_gap;
        }
    }
}

fn load_embedded_icon() -> Result<Icon> {
    let img = image::load_from_memory(ICON_BYTES)
        .context("falha ao decodificar icone")?
        .into_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).context("falha ao criar Icon")
}

fn center_position(event_loop: &ActiveEventLoop) -> Option<PhysicalPosition<i32>> {
    let monitor = event_loop
        .primary_monitor()
        .or_else(|| event_loop.available_monitors().next())?;
    let ms = monitor.size();
    let mp = monitor.position();
    let scale = monitor.scale_factor();
    let pw = (WINDOW_WIDTH * scale) as i32;
    let ph = (WINDOW_HEIGHT * scale) as i32;
    Some(PhysicalPosition::new(
        mp.x + (ms.width as i32 - pw) / 2,
        mp.y + (ms.height as i32 - ph) / 2,
    ))
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let icon = load_embedded_icon().ok();
        let mut attrs = WindowAttributes::default()
            .with_title("NeuroScan [Phase 5D — Brain 3D + Multi-Class Tumor WHO 2021]")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        if let Some(ic) = icon {
            attrs = attrs.with_window_icon(Some(ic));
        }

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                warn!(error = %err, "falha ao criar janela");
                event_loop.exit();
                return;
            }
        };

        if let Some(pos) = center_position(event_loop) {
            window.set_outer_position(pos);
        }
        info!(window_id = ?window.id(), "janela criada");

        let gpu = match pollster::block_on(GpuState::new(Arc::clone(&window))) {
            Ok(s) => s,
            Err(err) => {
                warn!(error = %err, "falha ao inicializar wgpu");
                event_loop.exit();
                return;
            }
        };
        self.gpu = Some(gpu);

        let device = &self.gpu.as_ref().unwrap().device;

        // Cerebro (obrigatorio)
        match Mesh::from_obj(device, BRAIN_OBJ) {
            Ok(m) => { info!("cerebro carregado"); self.meshes.push((m, BRAIN_COLOR)); }
            Err(e) => warn!(error = %e, "brain.obj nao encontrado"),
        }

        // Sub-regioes tumorais (todas opcionais — renderiza o que existir)
        let tumor_files = [
            (TUMOR_ET_OBJ,   ET_COLOR,   "ET"),
            (TUMOR_SNFH_OBJ, SNFH_COLOR, "SNFH"),
            (TUMOR_NETC_OBJ, NETC_COLOR, "NETC"),
        ];
        for (path, color, name) in &tumor_files {
            match Mesh::from_obj(device, path) {
                Ok(m) => { info!(class = name, "tumor carregado"); self.meshes.push((m, *color)); }
                Err(e) => warn!(error = %e, class = name, "tumor nao encontrado"),
            }
        }

        info!(
            total_meshes = self.meshes.len(),
            tumor_classes = self.meshes.len().saturating_sub(1),
            "scene carregada"
        );

        self.camera.target   = glam::Vec3::ZERO;
        self.camera.distance = 4.0;

        let size = window.inner_size();
        self.build_labels(size);
        self.last_frame = Instant::now();

        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("close solicitado");
                event_loop.exit();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if let Some(w) = &self.window {
                        w.set_cursor(if self.mouse_pressed {
                            CursorIcon::Grabbing
                        } else {
                            CursorIcon::Grab
                        });
                    }
                    if state == ElementState::Released {
                        self.mouse_pos = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((px, py)) = self.mouse_pos {
                        let dx = (position.x - px) as f32;
                        let dy = (position.y - py) as f32;
                        self.camera.yaw += dx * MOUSE_SENSITIVITY;
                        self.camera.pitch = (self.camera.pitch - dy * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((position.x, position.y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera.distance =
                    (self.camera.distance - scroll * ZOOM_SENSITIVITY).clamp(ZOOM_MIN, ZOOM_MAX);
            }

            WindowEvent::Resized(new_size) => {
                debug!(width = new_size.width, height = new_size.height, "resize");
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(new_size);
                }
                self.reposition_labels(new_size);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                self.last_frame = Instant::now();

                let w = self.gpu.as_ref().map_or(1280, |g| g.config.width);
                let h = self.gpu.as_ref().map_or(720,  |g| g.config.height);
                let cam = self.camera.build_uniform(w, h);

                if let Some(gpu) = &mut self.gpu {
                    let mesh_refs: Vec<(&Mesh, [f32; 3])> =
                        self.meshes.iter().map(|(m, c)| (m, *c)).collect();
                    let label_refs: Vec<&Label> = self.labels.iter().collect();
                    if let Err(e) = gpu.render(&cam, &mesh_refs, &label_refs) {
                        warn!(error = %e, "erro no render");
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("shutdown limpo");
    }
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!(version = env!("CARGO_PKG_VERSION"), "iniciando NeuroScan viewer");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
