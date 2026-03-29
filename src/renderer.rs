use anyhow::{Context, Result};
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

const SHADER_SOURCE: &str = include_str!("shader.wgsl");

/// Estado da pipeline grafica wgpu.
///
/// Criado uma vez apos a janela estar disponivel e vivo ate o shutdown.
/// `resize()` deve ser chamado em todo WindowEvent::Resized.
pub struct GpuState {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    size: PhysicalSize<u32>,
}

impl GpuState {
    /// Inicializa wgpu sobre a janela fornecida.
    ///
    /// O `Arc<Window>` e necessario para que a Surface tenha lifetime 'static
    /// sem depender do lifetime da janela — padrao recomendado com winit 0.30.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Safety: a surface referencia a janela via Arc — vive enquanto GpuState viver
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
            .request_device(
                &DeviceDescriptor {
                    label: Some("device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None, // trace path — None em producao
            )
            .await
            .context("falha ao criar device wgpu")?;

        let surface_caps = surface.get_capabilities(&adapter);

        // Preferir formato sRGB para output correto de cor
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

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            size,
        })
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

    /// Renderiza um frame — gradiente radial fullscreen.
    pub fn render(&self) -> Result<()> {
        let output = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Surface perdida (ex: minimize/restore): reconfigura e pula frame
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
                        // Clear com o mesmo grafite escuro do shader — sem flash branco
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

            render_pass.set_pipeline(&self.pipeline);
            // 6 vertices = 2 triangulos = quad fullscreen (sem vertex buffer)
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

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
