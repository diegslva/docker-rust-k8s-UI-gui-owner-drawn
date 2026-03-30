use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Uniform buffer enviado ao shader 3D a cada frame.
///
/// Layout deve ser identico ao struct CameraUniform em shader3d.wgsl.
/// Alinhamento: 160 bytes (multiplo de 16 para WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    /// Matriz Model * View * Projection — transforma vertices do espaco do objeto para clip space.
    pub mvp: [[f32; 4]; 4],
    /// Matriz normal (transposta inversa do model) — transforma normais corretamente.
    pub model_normal: [[f32; 4]; 4],
    /// Direcao da luz no espaco do mundo (normalizado).
    pub light_dir: [f32; 3],
    pub _pad: f32,
    /// Cor base da mesh (RGB). Sobrescrita pelo renderer por mesh para suportar multi-mesh.
    pub tint: [f32; 3],
    pub _pad2: f32,
}

/// Camera orbital: gira ao redor de um ponto alvo a distancia fixa.
///
/// yaw e pitch em radianos. Equivale ao ArcBall do 3D Slicer.
pub struct OrbitalCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl OrbitalCamera {
    pub fn new(distance: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            distance,
            yaw: 0.4,
            pitch: 0.3,
        }
    }

    /// Produz o CameraUniform para o frame atual.
    /// O campo `tint` e definido como zeros aqui — o renderer o sobrescreve
    /// por mesh antes de enviar para a GPU.
    pub fn build_uniform(&self, width: u32, height: u32) -> CameraUniform {
        let aspect = width as f32 / height.max(1) as f32;

        // Posicao do olho em coordenadas esfericas ao redor do target
        let eye = self.target
            + Vec3::new(
                self.yaw.cos() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.sin() * self.pitch.cos(),
            ) * self.distance;

        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45_f32.to_radians(), aspect, 0.01, 1000.0);
        let model = Mat4::IDENTITY;
        let mvp = proj * view * model;

        // Normal matrix: transposta da inversa do model (para iluminacao correta)
        let model_normal = model.inverse().transpose();

        // Luz vinda de cima e da frente levemente lateralizada
        let light_dir = Vec3::new(0.6, 1.0, 0.8).normalize();

        CameraUniform {
            mvp: mvp.to_cols_array_2d(),
            model_normal: model_normal.to_cols_array_2d(),
            light_dir: light_dir.to_array(),
            _pad: 0.0,
            tint: [0.0; 3], // sobrescrito pelo renderer por mesh
            _pad2: 0.0,
        }
    }
}
