use anyhow::{Context, Result};
use bytemuck::cast_slice;
use glyphon::{
    Cache, FontSystem, Resolution, SwashCache, TextArea, TextAtlas, TextRenderer, Viewport,
};
use std::sync::Arc;
use tracing::{debug, info};
use wgpu::{
    BindGroup, BindGroupLayout, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer,
    CommandEncoderDescriptor, DepthStencilState, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, LoadOp, Operations, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, RequestAdapterOptions,
    StoreOp, Surface, SurfaceConfiguration, Texture, TextureUsages, TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::camera::CameraUniform;
use crate::mesh::{Mesh, Vertex3D};
use crate::ui::Label;

const SHADER_2D: &str = include_str!("shader.wgsl");
const SHADER_3D: &str = include_str!("shader3d.wgsl");
const SHADER_2D_PRIM: &str = include_str!("shader2d_prim.wgsl");
const FONT_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.ttf");
const FONT_BOLD: &[u8] = include_bytes!("../assets/fonts/Inter-Bold.ttf");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const UNIFORM_ALIGN: usize = 256;
const MAX_MESHES: usize = 32;
const MAX_PRIM_VERTS: usize = 1024;
const MAX_PRIM_IDXS: usize = 2048;

// ---------------------------------------------------------------------------
// Primitivas 2D — geometria colorida em NDC (callout lines, boxes, separadores)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexPrim {
    pos: [f32; 2],
    col: [f32; 4],
}

/// Batch de geometria 2D colorida construido CPU-side a cada frame.
///
/// Coordenadas em pixels de tela; conversao para NDC e interna.
/// Suporta retangulos preenchidos e linhas como quads finos.
pub struct Prim2DBatch {
    verts: Vec<VertexPrim>,
    indices: Vec<u32>,
}

impl Prim2DBatch {
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Retangulo preenchido em coordenadas de pixel.
    #[allow(clippy::too_many_arguments)]
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, col: [f32; 4], sw: f32, sh: f32) {
        let tl = Self::px_to_ndc(x, y, sw, sh);
        let tr = Self::px_to_ndc(x + w, y, sw, sh);
        let br = Self::px_to_ndc(x + w, y + h, sw, sh);
        let bl = Self::px_to_ndc(x, y + h, sw, sh);
        self.push_quad([tl, tr, br, bl], col);
    }

    /// Linha renderizada como quad fino em coordenadas de pixel.
    #[allow(clippy::too_many_arguments)]
    pub fn line(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        col: [f32; 4],
        thickness: f32,
        sw: f32,
        sh: f32,
    ) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.5 {
            return;
        }
        // Normal perpendicular em pixel space
        let nx = -dy / len * thickness * 0.5;
        let ny = dx / len * thickness * 0.5;

        let a = Self::px_to_ndc(x1 + nx, y1 + ny, sw, sh);
        let b = Self::px_to_ndc(x1 - nx, y1 - ny, sw, sh);
        let c = Self::px_to_ndc(x2 - nx, y2 - ny, sw, sh);
        let d = Self::px_to_ndc(x2 + nx, y2 + ny, sw, sh);
        self.push_quad([a, b, c, d], col);
    }

    fn px_to_ndc(px: f32, py: f32, sw: f32, sh: f32) -> [f32; 2] {
        [(px / sw) * 2.0 - 1.0, 1.0 - (py / sh) * 2.0]
    }

    fn push_quad(&mut self, corners: [[f32; 2]; 4], col: [f32; 4]) {
        let base = self.verts.len() as u32;
        for c in &corners {
            self.verts.push(VertexPrim { pos: *c, col });
        }
        // 2 triangles: (0,1,2) (0,2,3)
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// ---------------------------------------------------------------------------

/// Propriedades de renderizacao por mesh.
pub struct MeshEntry<'a> {
    pub mesh: &'a Mesh,
    pub tint: [f32; 3],
    /// alpha: 1.0 = opaco, <1.0 = semi-transparente (alpha blend).
    pub alpha: f32,
}

pub struct GpuState {
    surface: Surface<'static>,
    pub device: Device,
    queue: Queue,
    pub config: SurfaceConfiguration,
    pipeline_2d: RenderPipeline,
    pipeline_2d_prim: RenderPipeline,
    /// Pipeline 3D opaca: depth write on, back-face cull, blend REPLACE.
    pipeline_3d_opaque: RenderPipeline,
    /// Pipeline 3D transparente: depth write OFF, back-face cull OFF, alpha blend.
    pipeline_3d_alpha: RenderPipeline,
    #[allow(dead_code)]
    camera_bind_group_layout: BindGroupLayout,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    /// Primitivas de cena (callout lines, boxes, separadores)
    prim_vert_buf: Buffer,
    prim_idx_buf: Buffer,
    /// Primitivas de overlay (menu bar + dropdown) — renderizadas APÓS o texto de cena
    overlay_prim_vert_buf: Buffer,
    overlay_prim_idx_buf: Buffer,
    depth_texture: Texture,
    size: PhysicalSize<u32>,
    // Texto de cena (callouts, título, painel lateral)
    font_system: FontSystem,
    swash_cache: SwashCache,
    #[allow(dead_code)]
    cache: Cache,
    viewport: Viewport,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    // Texto de overlay (itens do menu dropdown) — renderizado sobre o overlay de primitivas
    menu_text_atlas: TextAtlas,
    menu_text_renderer: TextRenderer,
}

impl GpuState {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("falha ao criar surface")?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("nenhum adaptador GPU encontrado")?;

        info!(backend = ?adapter.get_info().backend, name = adapter.get_info().name, "GPU");

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
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

        let pipeline_2d = Self::build_pipeline_2d(&device, &config);

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
            Self::create_depth_texture(&device, size.width.max(1), size.height.max(1));

        let pipeline_3d_opaque =
            Self::build_pipeline_3d(&device, &config, &camera_bind_group_layout, false);
        let pipeline_3d_alpha =
            Self::build_pipeline_3d(&device, &config, &camera_bind_group_layout, true);
        let pipeline_2d_prim = Self::build_pipeline_2d_prim(&device, &config);

        // Buffers dinâmicos para primitivas de overlay (menu bar + dropdown)
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

        // Renderer de cena (callouts, título, painel)
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

        // Renderer de overlay (itens do dropdown) — renderizado APÓS texto de cena
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
            Self::create_depth_texture(&self.device, new_size.width, new_size.height);
        debug!(width = new_size.width, height = new_size.height, "resize");
    }

    /// Renderiza um frame.
    ///
    /// Ordem de draw calls (dois render passes, mesmo encoder):
    ///
    /// Pass 1 — cena (LoadOp::Clear):
    ///   1. Gradiente 2D (fundo)
    ///   2. Meshes 3D opacas (tumores)
    ///   3. Meshes 3D transparentes (cérebro)
    ///   4. Primitivas de cena (callout lines, boxes, separadores)
    ///   5. Texto de cena (callouts, título, painel)
    ///
    /// Pass 2 — overlay (LoadOp::Load — preserva o frame do Pass 1):
    ///   6. Primitivas de overlay (menu bar + dropdown bg/hover)
    ///   7. Texto de overlay (itens do dropdown)
    ///
    /// Isso garante z-order correto: todo o menu fica acima de qualquer texto
    /// de cena, independente da ordem de construção dos labels.
    pub fn render(
        &mut self,
        camera_uniform: &CameraUniform,
        meshes: &[MeshEntry<'_>],
        labels: &[&Label],
        primitives: &Prim2DBatch,
        overlay_prims: &Prim2DBatch,
        overlay_labels: &[&Label],
    ) -> Result<()> {
        // ── Uniforms por mesh ──────────────────────────────────────────────────
        for (i, entry) in meshes.iter().enumerate() {
            debug_assert!(i < MAX_MESHES);
            let mut u = *camera_uniform;
            u.tint = entry.tint;
            u.alpha = entry.alpha;
            let u_arr = [u];
            let bytes = cast_slice::<CameraUniform, u8>(&u_arr);
            let mut slot = [0u8; UNIFORM_ALIGN];
            slot[..bytes.len()].copy_from_slice(bytes);
            self.queue
                .write_buffer(&self.camera_buffer, (i * UNIFORM_ALIGN) as u64, &slot);
        }

        // ── Upload primitivas de cena ──────────────────────────────────────────
        if !primitives.is_empty() {
            let vb_size = MAX_PRIM_VERTS * std::mem::size_of::<VertexPrim>();
            let ib_size = MAX_PRIM_IDXS * std::mem::size_of::<u32>();

            let vb_bytes = cast_slice::<VertexPrim, u8>(&primitives.verts);
            let mut vb_buf = vec![0u8; vb_size];
            let vb_len = vb_bytes.len().min(vb_size);
            vb_buf[..vb_len].copy_from_slice(&vb_bytes[..vb_len]);
            self.queue.write_buffer(&self.prim_vert_buf, 0, &vb_buf);

            let ib_bytes = cast_slice::<u32, u8>(&primitives.indices);
            let mut ib_buf = vec![0u8; ib_size];
            let ib_len = ib_bytes.len().min(ib_size);
            ib_buf[..ib_len].copy_from_slice(&ib_bytes[..ib_len]);
            self.queue.write_buffer(&self.prim_idx_buf, 0, &ib_buf);
        }

        // ── Upload primitivas de overlay ───────────────────────────────────────
        if !overlay_prims.is_empty() {
            let vb_size = MAX_PRIM_VERTS * std::mem::size_of::<VertexPrim>();
            let ib_size = MAX_PRIM_IDXS * std::mem::size_of::<u32>();

            let vb_bytes = cast_slice::<VertexPrim, u8>(&overlay_prims.verts);
            let mut vb_buf = vec![0u8; vb_size];
            let vb_len = vb_bytes.len().min(vb_size);
            vb_buf[..vb_len].copy_from_slice(&vb_bytes[..vb_len]);
            self.queue
                .write_buffer(&self.overlay_prim_vert_buf, 0, &vb_buf);

            let ib_bytes = cast_slice::<u32, u8>(&overlay_prims.indices);
            let mut ib_buf = vec![0u8; ib_size];
            let ib_len = ib_bytes.len().min(ib_size);
            ib_buf[..ib_len].copy_from_slice(&ib_bytes[..ib_len]);
            self.queue
                .write_buffer(&self.overlay_prim_idx_buf, 0, &ib_buf);
        }

        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );

        // ── Preparar texto de cena ─────────────────────────────────────────────
        let scene_areas: Vec<TextArea> = labels
            .iter()
            .map(|l| l.as_text_area(self.config.width, self.config.height))
            .collect();

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
                    scene_areas,
                    swash_cache,
                )
                .context("falha ao preparar texto de cena")?;
        }

        // ── Preparar texto de overlay ──────────────────────────────────────────
        let overlay_areas: Vec<TextArea> = overlay_labels
            .iter()
            .map(|l| l.as_text_area(self.config.width, self.config.height))
            .collect();

        {
            let GpuState {
                menu_text_renderer,
                device,
                queue,
                font_system,
                menu_text_atlas,
                viewport,
                swash_cache,
                ..
            } = self;
            menu_text_renderer
                .prepare(
                    device,
                    queue,
                    font_system,
                    menu_text_atlas,
                    viewport,
                    overlay_areas,
                    swash_cache,
                )
                .context("falha ao preparar texto de overlay")?;
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(err) => return Err(err).context("surface error"),
        };

        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());
        let depth_view = self
            .depth_texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("enc") });

        // ── Pass 1: cena (Clear) ───────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("scene_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.07,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Gradiente 2D (fundo)
            pass.set_pipeline(&self.pipeline_2d);
            pass.draw(0..6, 0..1);

            // 2. Meshes OPACAS (tumores)
            pass.set_pipeline(&self.pipeline_3d_opaque);
            for (i, entry) in meshes.iter().enumerate() {
                if entry.alpha < 1.0 {
                    continue;
                }
                pass.set_bind_group(0, &self.camera_bind_group, &[(i * UNIFORM_ALIGN) as u32]);
                pass.set_vertex_buffer(0, entry.mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(entry.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..entry.mesh.index_count, 0, 0..1);
            }

            // 3. Meshes TRANSPARENTES (cérebro)
            pass.set_pipeline(&self.pipeline_3d_alpha);
            for (i, entry) in meshes.iter().enumerate() {
                if entry.alpha >= 1.0 {
                    continue;
                }
                pass.set_bind_group(0, &self.camera_bind_group, &[(i * UNIFORM_ALIGN) as u32]);
                pass.set_vertex_buffer(0, entry.mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(entry.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..entry.mesh.index_count, 0, 0..1);
            }

            // 4. Primitivas de cena (callout lines + box backgrounds)
            if !primitives.is_empty() {
                pass.set_pipeline(&self.pipeline_2d_prim);
                pass.set_vertex_buffer(0, self.prim_vert_buf.slice(..));
                pass.set_index_buffer(self.prim_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..primitives.index_count() as u32, 0, 0..1);
            }

            // 5. Texto de cena (callouts, título, painel)
            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .context("falha ao renderizar texto de cena")?;
        }

        // ── Pass 2: overlay (Load — preserva o frame do Pass 1) ───────────────
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 6. Primitivas de overlay (menu bar + dropdown bg/hover)
            if !overlay_prims.is_empty() {
                pass.set_pipeline(&self.pipeline_2d_prim);
                pass.set_vertex_buffer(0, self.overlay_prim_vert_buf.slice(..));
                pass.set_index_buffer(
                    self.overlay_prim_idx_buf.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..overlay_prims.index_count() as u32, 0, 0..1);
            }

            // 7. Texto de overlay (itens do dropdown)
            self.menu_text_renderer
                .render(&self.menu_text_atlas, &self.viewport, &mut pass)
                .context("falha ao renderizar texto de overlay")?;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.text_atlas.trim();
        self.menu_text_atlas.trim();
        Ok(())
    }

    // --- Helpers ---

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn build_pipeline_2d(device: &Device, config: &SurfaceConfiguration) -> RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_2d"),
            source: wgpu::ShaderSource::Wgsl(SHADER_2D.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout_2d"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline_2d"),
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
                    blend: Some(BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    /// Pipeline para primitivas 2D coloridas com alpha blend.
    fn build_pipeline_2d_prim(device: &Device, config: &SurfaceConfiguration) -> RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_2d_prim"),
            source: wgpu::ShaderSource::Wgsl(SHADER_2D_PRIM.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout_2d_prim"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexPrim>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline_2d_prim"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vert_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // Sem depth test/write — primitivas 2D sempre sobre a cena 3D
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    /// `alpha_blend`: false = pipeline opaca (tumores), true = pipeline transparente (cerebro).
    fn build_pipeline_3d(
        device: &Device,
        config: &SurfaceConfiguration,
        camera_bgl: &BindGroupLayout,
        alpha_blend: bool,
    ) -> RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER_3D.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout_3d"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let blend = if alpha_blend {
            Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            })
        } else {
            Some(BlendState::REPLACE)
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(if alpha_blend {
                "pipeline_3d_alpha"
            } else {
                "pipeline_3d_opaque"
            }),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex3D::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: if alpha_blend {
                    None
                } else {
                    Some(wgpu::Face::Back)
                },
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: !alpha_blend,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
}
