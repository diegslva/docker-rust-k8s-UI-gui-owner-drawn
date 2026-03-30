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

/// Meshes NeuroScan: cerebro (FLAIR) + tumor (segmentacao BraTS)
const BRAIN_OBJ: &str = "assets/models/brain.obj";
const TUMOR_OBJ: &str = "assets/models/tumor.obj";

/// Cores das meshes (RGB linear)
/// Cerebro: substancia cinzenta (cinza rosado)
const BRAIN_COLOR: [f32; 3] = [0.82, 0.72, 0.70];
/// Tumor: vermelho-laranja vivo para contraste maximo
const TUMOR_COLOR: [f32; 3] = [0.90, 0.28, 0.18];

/// Sensibilidade do drag de mouse (radianos por pixel).
const MOUSE_SENSITIVITY: f32 = 0.005;
/// Sensibilidade do zoom via scroll.
const ZOOM_SENSITIVITY: f32 = 0.3;
/// Limites de distancia da camera.
const ZOOM_MIN: f32 = 1.5;
const ZOOM_MAX: f32 = 20.0;
/// Limite vertical da camera (evita gimbal lock no polo).
const PITCH_LIMIT: f32 = 1.5;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    /// Meshes com suas cores: (mesh, tint_rgb).
    /// Indice 0 = cerebro, indice 1 = tumor (se carregado).
    meshes: Vec<(Mesh, [f32; 3])>,
    camera: OrbitalCamera,
    labels: Vec<Label>,
    last_frame: Instant,
    // Estado do mouse
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

        let has_tumor = self.meshes.len() > 1;
        let subtitle = if has_tumor {
            "FLAIR + Tumor (BraTS)  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom"
        } else {
            "FLAIR  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom"
        };

        let mut title = Label::new_bold(
            fs,
            "NeuroScan  \u{00B7}  Cerebro 3D",
            32.0,
            Color::WHITE,
            0.0,
            0.0,
        );
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.06;

        let mut sub = Label::new(
            fs,
            subtitle,
            16.0,
            Color::rgb(140, 150, 165),
            0.0,
            0.0,
        );
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 6.0;

        self.labels = vec![title, sub];
    }

    fn reposition_labels(&mut self, size: PhysicalSize<u32>) {
        let w = size.width as f32;
        let h = size.height as f32;
        let title_y = h * 0.06;
        let title_lh = self.labels.first().map_or(32.0 * 1.25, |t| t.line_height());
        if let Some(title) = self.labels.get_mut(0) {
            title.x = (w - title.measured_width()) / 2.0;
            title.y = title_y;
        }
        if let Some(sub) = self.labels.get_mut(1) {
            sub.x = (w - sub.measured_width()) / 2.0;
            sub.y = title_y + title_lh + 6.0;
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
            .with_title("NeuroScan [Phase 5B — Brain 3D + Tumor]")
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

        // Carregar cerebro (obrigatorio)
        match Mesh::from_obj(device, BRAIN_OBJ) {
            Ok(m) => {
                info!("cerebro carregado");
                self.meshes.push((m, BRAIN_COLOR));
            }
            Err(err) => warn!(error = %err, "falha ao carregar brain.obj"),
        }

        // Carregar tumor (opcional — nao critico se ausente)
        match Mesh::from_obj(device, TUMOR_OBJ) {
            Ok(m) => {
                info!("tumor carregado");
                self.meshes.push((m, TUMOR_COLOR));
            }
            Err(err) => warn!(error = %err, "tumor.obj nao encontrado (apenas cerebro): {err}"),
        }

        info!(meshes = self.meshes.len(), "meshes carregadas");

        // Camera centralizada no mesh normalizado [-1, 1]
        self.camera.target = glam::Vec3::ZERO;
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

            // --- Mouse: botao pressionado / solto ---
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if let Some(w) = &self.window {
                        let cursor = if self.mouse_pressed {
                            CursorIcon::Grabbing
                        } else {
                            CursorIcon::Grab
                        };
                        w.set_cursor(cursor);
                    }
                    if state == ElementState::Released {
                        self.mouse_pos = None;
                    }
                }
            }

            // --- Mouse: movimento -- drag para rotacionar ---
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((prev_x, prev_y)) = self.mouse_pos {
                        let dx = (position.x - prev_x) as f32;
                        let dy = (position.y - prev_y) as f32;
                        self.camera.yaw += dx * MOUSE_SENSITIVITY;
                        self.camera.pitch = (self.camera.pitch - dy * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((position.x, position.y));
            }

            // --- Mouse: scroll -- zoom ---
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
                let now = Instant::now();
                self.last_frame = now;

                let w = self.gpu.as_ref().map_or(1280, |g| g.config.width);
                let h = self.gpu.as_ref().map_or(720, |g| g.config.height);
                let cam_uniform = self.camera.build_uniform(w, h);

                if let Some(gpu) = &mut self.gpu {
                    let mesh_refs: Vec<(&Mesh, [f32; 3])> =
                        self.meshes.iter().map(|(m, c)| (m, *c)).collect();
                    let label_refs: Vec<&Label> = self.labels.iter().collect();
                    if let Err(err) = gpu.render(&cam_uniform, &mesh_refs, &label_refs) {
                        warn!(error = %err, "erro no render");
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

    info!(version = env!("CARGO_PKG_VERSION"), "iniciando");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
