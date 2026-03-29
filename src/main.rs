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
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Icon, Window, WindowAttributes, WindowId};

const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8] = include_bytes!("../assets/icon_256x256.png");
const SKULL_OBJ: &str = "assets/models/skull.obj";

/// Velocidade de rotacao automatica do cranio (radianos por segundo).
const ROTATION_SPEED: f32 = 0.4;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    mesh: Option<Mesh>,
    camera: OrbitalCamera,
    labels: Vec<Label>,
    last_frame: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            mesh: None,
            camera: OrbitalCamera::new(5.0),
            labels: Vec::new(),
            last_frame: Instant::now(),
        }
    }

    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(
            fs,
            "Fase 4A  \u{00B7}  3D Cranio",
            32.0,
            Color::WHITE,
            0.0,
            0.0,
        );
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.06;

        let mut sub = Label::new(
            fs,
            "Phong shading  \u{00B7}  Depth buffer  \u{00B7}  OBJ loader",
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
            .with_title("docker-rust-k8s-ui-gui [Phase 4A — 3D Skull]")
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

        // Carregar mesh do cranio
        let device = &self.gpu.as_ref().unwrap().device;
        match Mesh::from_obj(device, SKULL_OBJ) {
            Ok(m) => {
                info!("cranio carregado com sucesso");
                self.mesh = Some(m);
            }
            Err(err) => warn!(error = %err, "falha ao carregar skull.obj"),
        }

        // Camera centrada no cranio (bounding center aproximado do modelo Blender)
        self.camera.target = glam::Vec3::new(0.0, 2.5, 0.0);
        self.camera.distance = 5.0;

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
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Auto-rotacao suave ao redor do cranio
                self.camera.yaw += ROTATION_SPEED * dt;

                let w = self.gpu.as_ref().map_or(1280, |g| g.config.width);
                let h = self.gpu.as_ref().map_or(720, |g| g.config.height);
                let cam_uniform = self.camera.build_uniform(w, h);

                if let (Some(gpu), Some(mesh)) = (&mut self.gpu, &self.mesh) {
                    let label_refs: Vec<&Label> = self.labels.iter().collect();
                    if let Err(err) = gpu.render(&cam_uniform, mesh, &label_refs) {
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
