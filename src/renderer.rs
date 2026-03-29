use anyhow::{Context, Result};
use glyphon::{
    Cache, FontSystem, Resolution, SwashCache, TextArea, TextAtlas, TextRenderer, Viewport,
};
use std::sync::Arc;
use tracing::{debug, info};
use wgpu::{
    CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    Limits, LoadOp, Operations, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RequestAdapterOptions, StoreOp, Surface, SurfaceConfiguration, TextureUsages,
    TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::ui::Label;

const SHADER_SOURCE: &str = include_str!("shader.wgsl");
const FONT_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.ttf");
const FONT_BOLD: &[u8] = include_bytes!("../assets/fonts/Inter-Bold.ttf");

/// Estado da pipeline grafica wgpu com sistema de texto glyphon integrado.
///
/// Responsabilidade: GPU, surface, pipeline de gradiente, atlas de texto.
/// Nao conhece o conteudo da tela — recebe os Labels prontos em `render()`.
pub struct GpuState {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    size: PhysicalSize<u32>,
    font_system: FontSystem,
    swash_cache: SwashCache,
    #[allow(dead_code)] // mantido vivo: referenciado por TextAtlas e Viewport internamente
    cache: Cache,
    viewport: Viewport,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
}

impl GpuState {
    /// Inicializa wgpu e o sistema de texto glyphon sobre a janela fornecida.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("falha ao criar surface wgpu")?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("nenhum adaptador GPU compativel encontrado")?;

        info!(
            backend = ?adapter.get_info().backend,
            name = adapter.get_info().name,
            "adaptador GPU selecionado"
        );

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .context("falha ao criar device wgpu")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        debug!(format = ?surface_format, "formato da surface selecionado");

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let pipeline = Self::build_pipeline(&device, &config);

        // Carrega Inter Regular e Bold embutidas no binario
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT_REGULAR.to_vec());
        font_system.db_mut().load_font_data(FONT_BOLD.to_vec());

        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        info!("sistema de texto inicializado com Inter Regular + Bold");

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            size,
            font_system,
            swash_cache,
            cache,
            viewport,
            text_atlas,
            text_renderer,
        })
    }

    /// Acesso ao FontSystem para criacao de Labels fora do renderer.
    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    /// Tamanho atual da surface em pixels fisicos. Usado pelo App para layout inicial dos Labels.
    #[allow(dead_code)]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    /// Reconstroi a surface apos redimensionamento da janela.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        debug!(
            width = new_size.width,
            height = new_size.height,
            "surface reconfigurada"
        );
    }

    /// Renderiza um frame: gradiente radial + todos os Labels fornecidos.
    ///
    /// Os Labels sao posicionados e coloridos por quem os criou (a tela/app).
    /// O renderer apenas os converte em TextAreas e passa ao glyphon.
    pub fn render(&mut self, labels: &[&Label]) -> Result<()> {
        // Atualizar resolucao do viewport glyphon — necessario todo frame
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );

        // Converter cada Label em TextArea (operacao barata, sem alocacao de GPU)
        let text_areas: Vec<TextArea> = labels
            .iter()
            .map(|l| l.as_text_area(self.config.width, self.config.height))
            .collect();

        // Preparar texto para GPU — borrow split explicito para satisfazer o borrow checker
        {
            let GpuState {
                text_renderer,
                device,
                queue,
                font_system,
                text_atlas,
                viewport,
                swash_cache,
                ..
            } = self;

            text_renderer
                .prepare(
                    device,
                    queue,
                    font_system,
                    text_atlas,
                    viewport,
                    text_areas,
                    swash_cache,
                )
                .context("falha ao preparar texto para GPU")?;
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(err) => return Err(err).context("falha ao obter texture da surface"),
        };

        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.118,
                            g: 0.118,
                            b: 0.141,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Gradiente radial fullscreen
            render_pass.set_pipeline(&self.pipeline);
            render_pass.draw(0..6, 0..1);

            // 2. Todos os Labels sobre o gradiente
            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut render_pass)
                .context("falha ao renderizar texto")?;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.text_atlas.trim();

        Ok(())
    }

    fn build_pipeline(device: &Device, config: &SurfaceConfiguration) -> RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }
}
