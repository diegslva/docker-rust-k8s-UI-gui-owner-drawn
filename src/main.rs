mod app;
mod camera;
mod mesh;
mod pipeline;
mod renderer;
mod ui;

use anyhow::Result;
use tracing_subscriber::{EnvFilter, fmt};
use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .init();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "NeuroScan viewer iniciando"
    );

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
