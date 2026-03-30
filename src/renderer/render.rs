use anyhow::{Context, Result};
use bytemuck::cast_slice;
use glyphon::{Resolution, TextArea};
use wgpu::{
    CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TextureViewDescriptor,
};

use crate::camera::CameraUniform;
use crate::ui::Label;

use super::prim2d::{MeshEntry, Prim2DBatch, VertexPrim};
use super::state::{GpuState, MAX_MESHES, MAX_PRIM_IDXS, MAX_PRIM_VERTS, UNIFORM_ALIGN};

impl GpuState {
    /// Renderiza um frame.
    ///
    /// Ordem de draw calls (dois render passes, mesmo encoder):
    ///
    /// Pass 1 — cena (LoadOp::Clear):
    ///   1. Gradiente 2D (fundo)
    ///   2. Meshes 3D opacas (tumores)
    ///   3. Meshes 3D transparentes (cerebro)
    ///   4. Primitivas de cena (callout lines, boxes, separadores)
    ///   5. Texto de cena (callouts, titulo, painel)
    ///
    /// Pass 2 — overlay (LoadOp::Load — preserva o frame do Pass 1):
    ///   6. Primitivas de overlay (menu bar + dropdown bg/hover)
    ///   7. Texto de overlay (itens do dropdown)
    ///
    /// Isso garante z-order correto: todo o menu fica acima de qualquer texto
    /// de cena, independente da ordem de construcao dos labels.
    pub fn render(
        &mut self,
        camera_uniform: &CameraUniform,
        meshes: &[MeshEntry<'_>],
        labels: &[&Label],
        primitives: &Prim2DBatch,
        overlay_prims: &Prim2DBatch,
        overlay_labels: &[&Label],
    ) -> Result<()> {
        // -- Uniforms por mesh -----------------------------------------------
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

        // -- Upload primitivas de cena ----------------------------------------
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

        // -- Upload primitivas de overlay -------------------------------------
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

        // -- Preparar texto de cena -------------------------------------------
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

        // -- Preparar texto de overlay ----------------------------------------
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

        // -- Pass 1: cena (Clear) ---------------------------------------------
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

            // 3. Meshes TRANSPARENTES (cerebro)
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

            // 5. Texto de cena (callouts, titulo, painel)
            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .context("falha ao renderizar texto de cena")?;
        }

        // -- Pass 2: overlay (Load — preserva o frame do Pass 1) -------------
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
}
