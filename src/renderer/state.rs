use anyhow::{Context, Result};
use glyphon::{Cache, FontSystem, SwashCache, TextAtlas, TextRenderer, Viewport};
use std::sync::Arc;
use tracing::{debug, info};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, DepthStencilState, Device, Queue, RenderPipeline, Surface,
    SurfaceConfiguration, Texture, TextureUsages,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::camera::CameraUniform;

use super::pipelines::{self, DEPTH_FORMAT, FONT_BOLD, FONT_REGULAR};
use super::prim2d::VertexPrim;

pub(crate) const UNIFORM_ALIGN: usize = 256;
pub(crate) const MAX_MESHES: usize = 32;
pub(crate) const MAX_PRIM_VERTS: usize = 1024;
pub(crate) const MAX_PRIM_IDXS: usize = 2048;

pub struct GpuState {
    pub(crate) surface: Surface<'static>,
    pub device: Device,
    pub(crate) queue: Queue,
    pub config: SurfaceConfiguration,
    pub(crate) pipeline_2d: RenderPipeline,
    pub(crate) pipeline_2d_prim: RenderPipeline,
    /// Pipeline 3D opaca: depth write on, back-face cull, blend REPLACE.
    pub(crate) pipeline_3d_opaque: RenderPipeline,
    /// Pipeline 3D transparente: depth write OFF, back-face cull OFF, alpha blend.
    pub(crate) pipeline_3d_alpha: RenderPipeline,
    #[allow(dead_code)]
    pub(crate) camera_bind_group_layout: BindGroupLayout,
    pub(crate) camera_buffer: Buffer,
    pub(crate) camera_bind_group: BindGroup,
    /// Primitivas de cena (callout lines, boxes, separadores)
    pub(crate) prim_vert_buf: Buffer,
    pub(crate) prim_idx_buf: Buffer,
    /// Primitivas de overlay (menu bar + dropdown) — renderizadas APOS o texto de cena
    pub(crate) overlay_prim_vert_buf: Buffer,
    pub(crate) overlay_prim_idx_buf: Buffer,
    pub(crate) depth_texture: Texture,
    pub(crate) size: PhysicalSize<u32>,
    // Texto de cena (callouts, titulo, painel lateral)
    pub(crate) font_system: FontSystem,
    pub(crate) swash_cache: SwashCache,
    #[allow(dead_code)]
    pub(crate) cache: Cache,
    pub(crate) viewport: Viewport,
    pub(crate) text_atlas: TextAtlas,
    pub(crate) text_renderer: TextRenderer,
    // Texto de overlay (itens do menu dropdown) — renderizado sobre o overlay de primitivas
    pub(crate) menu_text_atlas: TextAtlas,
    pub(crate) menu_text_renderer: TextRenderer,
}

impl GpuState {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("falha ao criar surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("nenhum adaptador GPU encontrado")?;

        info!(backend = ?adapter.get_info().backend, name = adapter.get_info().name, "GPU");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .context("falha ao criar device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        debug!(format = ?surface_format, "surface format");

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

        let pipeline_2d = pipelines::build_pipeline_2d(&device, &config);

        // Bind group layout com dynamic offset
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CameraUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: (MAX_MESHES * UNIFORM_ALIGN) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<CameraUniform>() as u64),
                }),
            }],
        });

        // Buffers dinamicos para primitivas 2D
        let prim_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prim_vert_buf"),
            size: (MAX_PRIM_VERTS * std::mem::size_of::<VertexPrim>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prim_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prim_idx_buf"),
            size: (MAX_PRIM_IDXS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_texture =
            pipelines::create_depth_texture(&device, size.width.max(1), size.height.max(1));

        let pipeline_3d_opaque =
            pipelines::build_pipeline_3d(&device, &config, &camera_bind_group_layout, false);
        let pipeline_3d_alpha =
            pipelines::build_pipeline_3d(&device, &config, &camera_bind_group_layout, true);
        let pipeline_2d_prim = pipelines::build_pipeline_2d_prim(&device, &config);

        // Buffers dinamicos para primitivas de overlay (menu bar + dropdown)
        let overlay_prim_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_prim_vert_buf"),
            size: (MAX_PRIM_VERTS * std::mem::size_of::<VertexPrim>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_prim_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_prim_idx_buf"),
            size: (MAX_PRIM_IDXS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT_REGULAR.to_vec());
        font_system.db_mut().load_font_data(FONT_BOLD.to_vec());

        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);

        // Renderer de cena (callouts, titulo, painel)
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );

        // Renderer de overlay (itens do dropdown) — renderizado APOS texto de cena
        let mut menu_text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let menu_text_renderer = TextRenderer::new(
            &mut menu_text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );

        info!("GpuState: pipeline 2D + 3D opaque + 3D alpha + prim2D + texto + overlay");

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline_2d,
            pipeline_2d_prim,
            pipeline_3d_opaque,
            pipeline_3d_alpha,
            camera_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            prim_vert_buf,
            prim_idx_buf,
            overlay_prim_vert_buf,
            overlay_prim_idx_buf,
            depth_texture,
            size,
            font_system,
            swash_cache,
            cache,
            viewport,
            text_atlas,
            text_renderer,
            menu_text_atlas,
            menu_text_renderer,
        })
    }

    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    #[allow(dead_code)]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            pipelines::create_depth_texture(&self.device, new_size.width, new_size.height);
        debug!(width = new_size.width, height = new_size.height, "resize");
    }
}
