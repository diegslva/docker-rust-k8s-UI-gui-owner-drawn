use anyhow::Result;
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

struct App {
    window: Option<Window>,
}

impl App {
    fn new() -> Self {
        Self { window: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard: em desktop resumed() dispara uma vez, mas em mobile pode repetir
        if self.window.is_some() {
            debug!("resumed() chamado com janela ja existente, ignorando");
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title("docker-rust-k8s-ui-gui [Phase 0]")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        match event_loop.create_window(attributes) {
            Ok(window) => {
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
