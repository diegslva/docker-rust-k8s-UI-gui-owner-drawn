use anyhow::{Context, Result};
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Icon, Window, WindowAttributes, WindowId};

const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const ICON_BYTES: &[u8] = include_bytes!("../assets/icon_256x256.png");

struct App {
    window: Option<Window>,
}

impl App {
    fn new() -> Self {
        Self { window: None }
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

    // Converter tamanho logico da janela para fisico no scale do monitor
    let window_physical_width = (WINDOW_WIDTH * scale) as i32;
    let window_physical_height = (WINDOW_HEIGHT * scale) as i32;

    let x = monitor_pos.x + (monitor_size.width as i32 - window_physical_width) / 2;
    let y = monitor_pos.y + (monitor_size.height as i32 - window_physical_height) / 2;

    Some(PhysicalPosition::new(x, y))
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard: em desktop resumed() dispara uma vez, mas em mobile pode repetir
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
            .with_title("docker-rust-k8s-ui-gui [Phase 0]")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

        if let Some(icon) = icon {
            attributes = attributes.with_window_icon(Some(icon));
        }

        match event_loop.create_window(attributes) {
            Ok(window) => {
                // Centralizar apos criacao — posicao depende do monitor real
                if let Some(pos) = center_position(event_loop) {
                    window.set_outer_position(pos);
                    debug!(x = pos.x, y = pos.y, "janela centralizada no monitor");
                }

                info!(window_id = ?window.id(), "janela criada com sucesso");
                self.window = Some(window);
            }
            Err(err) => {
                warn!(error = %err, "falha ao criar janela, encerrando");
                event_loop.exit();
            }
        }
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
            WindowEvent::Resized(size) => {
                debug!(
                    width = size.width,
                    height = size.height,
                    "janela redimensionada"
                );
            }
            WindowEvent::RedrawRequested => {
                // Phase 0: sem rendering, mas mantemos o contrato do event loop
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Cleanup vai aqui — run_app() nao retorna em todas as plataformas (macOS)
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
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
