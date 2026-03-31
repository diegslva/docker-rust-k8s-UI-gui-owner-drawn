use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Uniform buffer enviado ao shader 3D a cada frame.
/// Layout identico ao struct CameraUniform em shader3d.wgsl.
/// Tamanho: 176 bytes (multiplo de 16, conforme requerimento WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub mvp: [[f32; 4]; 4],          // 64 bytes
    pub model_normal: [[f32; 4]; 4], // 64 bytes
    pub light_dir: [f32; 3],         // 12 bytes
    pub roughness: f32,              //  4 bytes (0.0=espelho, 1.0=fosco)
    pub tint: [f32; 3],              // 12 bytes
    pub alpha: f32,                  //  4 bytes (1.0=opaco, <1.0=transparente)
    pub sss_strength: f32,           //  4 bytes (subsurface scattering, 0.0=off)
    pub _pad: [f32; 3],              // 12 bytes padding (total: 176)
}

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

    /// Produz o CameraUniform base para o frame atual.
    /// tint e alpha sao sobrescritos pelo renderer por mesh.
    pub fn build_uniform(&self, width: u32, height: u32) -> CameraUniform {
        let aspect = width as f32 / height.max(1) as f32;

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

        let model_normal = model.inverse().transpose();
        let light_dir = Vec3::new(0.6, 1.0, 0.8).normalize();

        CameraUniform {
            mvp: mvp.to_cols_array_2d(),
            model_normal: model_normal.to_cols_array_2d(),
            light_dir: light_dir.to_array(),
            roughness: 0.5,    // sobrescrito pelo renderer
            tint: [0.0; 3],    // sobrescrito pelo renderer
            alpha: 1.0,        // sobrescrito pelo renderer
            sss_strength: 0.0, // sobrescrito pelo renderer
            _pad: [0.0; 3],
        }
    }
}
