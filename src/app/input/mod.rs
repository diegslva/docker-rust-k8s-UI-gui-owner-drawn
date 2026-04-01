mod events;
mod redraw;

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Icon, WindowAttributes, WindowId};

use crate::renderer::{GpuState, Prim2DBatch};
use crate::ui::Label;

use super::state::{App, ICON_BYTES, WINDOW_HEIGHT, WINDOW_WIDTH, load_brain_meshes_bg};

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
            .with_title("NeuroScan AI — Segmentação Cerebral Inteligente")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_visible(false);
        if let Some(ic) = icon {
            attrs = attrs.with_window_icon(Some(ic));
        }

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                warn!(error = %e, "falha ao criar janela");
                event_loop.exit();
                return;
            }
        };
        if let Some(pos) = center_position(event_loop) {
            window.set_outer_position(pos);
        }

        let gpu = match pollster::block_on(GpuState::new(Arc::clone(&window))) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "falha ao inicializar wgpu");
                event_loop.exit();
                return;
            }
        };
        self.gpu = Some(gpu);

        let size = window.inner_size();
        self.build_splash_labels(size);

        // Carregar todos os meshes em thread de fundo
        // wgpu::Device is internally Arc<..> — clone is cheap
        use neuroscan_core::TOP_CASES;
        let device = self.gpu.as_ref().unwrap().device.clone();
        self.splash_rx = Some(load_brain_meshes_bg(device, TOP_CASES[0]));
        info!("splash iniciada — carregamento de meshes em background");

        self.camera.target = glam::Vec3::ZERO;
        self.camera.distance = 4.0;
        self.last_frame = std::time::Instant::now();
        self.last_interaction = std::time::Instant::now();

        // Renderizar um frame escuro ANTES de mostrar a janela — elimina flash branco do OS.
        self.window = Some(Arc::clone(&window));
        {
            let cam = self.camera.build_uniform(size.width, size.height);
            let sw = size.width as f32;
            let sh = size.height as f32;
            let mut first_prims = Prim2DBatch::new();
            first_prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, 1.0], sw, sh);
            let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
            if let Some(gpu) = &mut self.gpu {
                let empty_overlay = Prim2DBatch::new();
                if let Err(e) = gpu.render(
                    &cam,
                    &[],
                    &label_refs,
                    &first_prims,
                    &empty_overlay,
                    &[],
                    false,
                ) {
                    warn!(error = %e, "erro no frame inicial da splash");
                }
            }
        }
        window.set_visible(true);
        self.window_shown = true;
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.window_event_inner(event_loop, event);
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        info!("shutdown");
    }
}
