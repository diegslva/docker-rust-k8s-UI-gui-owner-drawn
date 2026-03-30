use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Device, util::DeviceExt};

/// Vertice 3D com posicao e normal — layout exato do vertex buffer na GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex3D {
    /// Layout do vertex buffer para a pipeline 3D.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex3D>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

/// Mesh carregada na GPU: vertex buffer + index buffer + centroide 3D.
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    /// Centroide geometrico em espaco de objeto (media de todos os vertices).
    pub centroid: glam::Vec3,
}

impl Mesh {
    /// Carrega um arquivo OBJ do disco e faz upload para a GPU.
    ///
    /// Triangula automaticamente. Calcula normais por face se o OBJ nao tiver.
    pub fn from_obj(device: &Device, path: &str) -> Result<Self> {
        let (models, _) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .context("falha ao carregar OBJ")?;

        let mut vertices: Vec<Vertex3D> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut index_offset: u32 = 0;

        for model in &models {
            let mesh = &model.mesh;
            let vertex_count = mesh.positions.len() / 3;

            // Calcular normais por face se ausentes no OBJ
            let has_normals = mesh.normals.len() == mesh.positions.len();

            for i in 0..vertex_count {
                let pos = [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                ];
                let normal = if has_normals {
                    [
                        mesh.normals[i * 3],
                        mesh.normals[i * 3 + 1],
                        mesh.normals[i * 3 + 2],
                    ]
                } else {
                    [0.0, 1.0, 0.0] // sera recalculado abaixo
                };
                vertices.push(Vertex3D {
                    position: pos,
                    normal,
                });
            }

            for &idx in &mesh.indices {
                indices.push(index_offset + idx);
            }

            // Recalcular normais por face se o OBJ nao as forneceu
            if !has_normals {
                let base = index_offset as usize;
                let face_count = mesh.indices.len() / 3;
                for f in 0..face_count {
                    let i0 = base + mesh.indices[f * 3] as usize;
                    let i1 = base + mesh.indices[f * 3 + 1] as usize;
                    let i2 = base + mesh.indices[f * 3 + 2] as usize;
                    let p0 = glam::Vec3::from(vertices[i0].position);
                    let p1 = glam::Vec3::from(vertices[i1].position);
                    let p2 = glam::Vec3::from(vertices[i2].position);
                    let n = (p1 - p0).cross(p2 - p0).normalize();
                    vertices[i0].normal = n.to_array();
                    vertices[i1].normal = n.to_array();
                    vertices[i2].normal = n.to_array();
                }
            }

            index_offset += vertex_count as u32;
        }

        let centroid = if vertices.is_empty() {
            glam::Vec3::ZERO
        } else {
            let sum = vertices
                .iter()
                .map(|v| glam::Vec3::from(v.position))
                .fold(glam::Vec3::ZERO, |acc, v| acc + v);
            sum / vertices.len() as f32
        };

        tracing::info!(
            vertices = vertices.len(),
            triangles = indices.len() / 3,
            "mesh carregada"
        );

        Self::upload(device, &vertices, &indices, centroid)
    }

    fn upload(
        device: &Device,
        vertices: &[Vertex3D],
        indices: &[u32],
        centroid: glam::Vec3,
    ) -> Result<Self> {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            centroid,
        })
    }
}
