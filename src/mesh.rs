use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Device, util::DeviceExt};

/// Vertice 3D com posicao, normal e UV — layout exato do vertex buffer na GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
}

impl Vertex3D {
    /// Layout do vertex buffer para a pipeline 3D.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex3D>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
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

        Self::from_models(device, &models)
    }

    /// Carrega um OBJ a partir de bytes em memoria (rust-embed) e faz upload para a GPU.
    pub fn from_obj_bytes(device: &Device, data: &[u8]) -> Result<Self> {
        let mut reader = std::io::BufReader::new(std::io::Cursor::new(data));
        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| Ok(Default::default()),
        )
        .context("falha ao carregar OBJ de bytes")?;

        Self::from_models(device, &models)
    }

    /// Constroi vertices e indices a partir de modelos tobj.
    fn from_models(device: &Device, models: &[tobj::Model]) -> Result<Self> {
        let mut vertices: Vec<Vertex3D> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut index_offset: u32 = 0;

        for model in models {
            let mesh = &model.mesh;
            let vertex_count = mesh.positions.len() / 3;

            // Calcular normais por face se ausentes no OBJ
            let has_normals = mesh.normals.len() == mesh.positions.len();
            let has_uvs = mesh.texcoords.len() >= vertex_count * 2;

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
                let texcoord = if has_uvs {
                    [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
                } else {
                    [0.0, 0.0]
                };
                vertices.push(Vertex3D {
                    position: pos,
                    normal,
                    texcoord,
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

/// Gera vertices e indices de um quad de corte MRI posicionado no espaco 3D.
///
/// O quad e alinhado ao plano anatomico escolhido, com UVs [0,1] para texture sampling.
pub fn generate_slice_quad(
    plane: crate::volume::SlicePlane,
    position: f32,
    world_min: glam::Vec3,
    world_max: glam::Vec3,
) -> ([Vertex3D; 4], [u32; 6]) {
    use crate::volume::SlicePlane;

    let (v0, v1, v2, v3, normal) = match plane {
        SlicePlane::Axial => {
            // XY plane, varies Z
            let z = world_min.z + position * (world_max.z - world_min.z);
            (
                glam::Vec3::new(world_min.x, world_min.y, z),
                glam::Vec3::new(world_max.x, world_min.y, z),
                glam::Vec3::new(world_max.x, world_max.y, z),
                glam::Vec3::new(world_min.x, world_max.y, z),
                [0.0, 0.0, 1.0],
            )
        }
        SlicePlane::Coronal => {
            // XZ plane, varies Y
            let y = world_min.y + position * (world_max.y - world_min.y);
            (
                glam::Vec3::new(world_min.x, y, world_min.z),
                glam::Vec3::new(world_max.x, y, world_min.z),
                glam::Vec3::new(world_max.x, y, world_max.z),
                glam::Vec3::new(world_min.x, y, world_max.z),
                [0.0, 1.0, 0.0],
            )
        }
        SlicePlane::Sagittal => {
            // YZ plane, varies X
            let x = world_min.x + position * (world_max.x - world_min.x);
            (
                glam::Vec3::new(x, world_min.y, world_min.z),
                glam::Vec3::new(x, world_max.y, world_min.z),
                glam::Vec3::new(x, world_max.y, world_max.z),
                glam::Vec3::new(x, world_min.y, world_max.z),
                [1.0, 0.0, 0.0],
            )
        }
    };

    let vertices = [
        Vertex3D {
            position: v0.to_array(),
            normal,
            texcoord: [0.0, 0.0],
        },
        Vertex3D {
            position: v1.to_array(),
            normal,
            texcoord: [1.0, 0.0],
        },
        Vertex3D {
            position: v2.to_array(),
            normal,
            texcoord: [1.0, 1.0],
        },
        Vertex3D {
            position: v3.to_array(),
            normal,
            texcoord: [0.0, 1.0],
        },
    ];
    let indices = [0u32, 1, 2, 0, 2, 3];

    (vertices, indices)
}
