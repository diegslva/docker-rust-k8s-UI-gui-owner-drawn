mod renderer;
mod ui;

use anyhow::{Context, Result};
use renderer::GpuState;
use std::sync::Arc;
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

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    // Labels da tela — equivalente a ter TLabel declarados no Form do Delphi
    title: Option<Label>,
    subtitle: Option<Label>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            title: None,
            subtitle: None,
        }
    }

    /// Cria os Labels e calcula posicao inicial com base no tamanho da janela.
    /// Chamado apos GpuState estar pronto (FontSystem disponivel).
    fn build_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;

        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(fs, "docker-rust-k8s-ui-gui", 52.0, Color::WHITE, 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.40;

        let mut subtitle = Label::new(
            fs,
            "Phase 3 \u{00B7} Label como componente",
            22.0,
            Color::rgb(180, 185, 200),
            0.0,
            0.0,
        );
        subtitle.x = (w - subtitle.measured_width()) / 2.0;
        subtitle.y = title.y + title.line_height() + 16.0;

        self.title = Some(title);
        self.subtitle = Some(subtitle);
    }

    /// Recalcula posicao dos Labels apos redimensionamento.
    fn reposition_labels(&mut self, size: PhysicalSize<u32>) {
        let w = size.width as f32;
        let h = size.height as f32;

        if let Some(title) = &mut self.title {
            title.x = (w - title.measured_width()) / 2.0;
            title.y = h * 0.40;
        }
        if let Some(subtitle) = &mut self.subtitle {
            let title_top = self.title.as_ref().map_or(h * 0.40, |t| t.y);
            let title_lh = self.title.as_ref().map_or(52.0 * 1.25, |t| t.line_height());
            subtitle.x = (w - subtitle.measured_width()) / 2.0;
            subtitle.y = title_top + title_lh + 16.0;
        }
    }
}

/// Decodifica o PNG embutido em RGBA e cria o Icon do winit.
fn load_embedded_icon() -> Result<Icon> {
    let img = image::load_from_memory(ICON_BYTES)
        .context("falha ao decodificar PNG do icone embutido")?
        .into_rgba8();

    let (width, height) = img.dimensions();
    let rgba = img.into_raw();

    Icon::from_rgba(rgba, width, height).context("falha ao criar Icon a partir dos pixels RGBA")
}

/// Calcula posicao para centralizar a janela no monitor.
fn center_position(event_loop: &ActiveEventLoop) -> Option<PhysicalPosition<i32>> {
    let monitor = event_loop
        .primary_monitor()
        .or_else(|| event_loop.available_monitors().next())?;

    let monitor_size = monitor.size();
    let monitor_pos = monitor.position();
    let scale = monitor.scale_factor();

    let window_physical_width = (WINDOW_WIDTH * scale) as i32;
    let window_physical_height = (WINDOW_HEIGHT * scale) as i32;

    let x = monitor_pos.x + (monitor_size.width as i32 - window_physical_width) / 2;
    let y = monitor_pos.y + (monitor_size.height as i32 - window_physical_height) / 2;

    Some(PhysicalPosition::new(x, y))
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            debug!("resumed() chamado com janela ja existente, ignorando");
            return;
        }

        let icon = match load_embedded_icon() {
            Ok(icon) => {
                debug!("icone carregado com sucesso");
                Some(icon)
            }
            Err(err) => {
                warn!(error = %err, "falha ao carregar icone, continuando sem icone");
                None
            }
        };

        let mut attributes = WindowAttributes::default()
            .with_title("docker-rust-k8s-ui-gui [Phase 3]")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

        if let Some(icon) = icon {
            attributes = attributes.with_window_icon(Some(icon));
        }

        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                warn!(error = %err, "falha ao criar janela, encerrando");
                event_loop.exit();
                return;
            }
        };

        if let Some(pos) = center_position(event_loop) {
            window.set_outer_position(pos);
            debug!(x = pos.x, y = pos.y, "janela centralizada no monitor");
        }

        info!(window_id = ?window.id(), "janela criada com sucesso");

        let gpu = match pollster::block_on(GpuState::new(Arc::clone(&window))) {
            Ok(state) => state,
            Err(err) => {
                warn!(error = %err, "falha ao inicializar wgpu, encerrando");
                event_loop.exit();
                return;
            }
        };

        self.gpu = Some(gpu);

        // Labels criados apos GpuState (FontSystem necessario para shaping)
        let size = window.inner_size();
        self.build_labels(size);

        info!("pipeline wgpu e labels inicializados com sucesso");

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
                info!("close solicitado pelo usuario");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                debug!(
                    width = new_size.width,
                    height = new_size.height,
                    "janela redimensionada"
                );
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(new_size);
                }
                self.reposition_labels(new_size);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Coletar refs dos labels existentes (campos distintos de gpu — borrow seguro)
                let labels: Vec<&Label> = [self.title.as_ref(), self.subtitle.as_ref()]
                    .into_iter()
                    .flatten()
                    .collect();

                if let Some(gpu) = &mut self.gpu {
                    if let Err(err) = gpu.render(&labels) {
                        warn!(error = %err, "erro durante render, pulando frame");
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("shutdown limpo concluido");
    }
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "iniciando docker-rust-k8s-ui-gui"
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
