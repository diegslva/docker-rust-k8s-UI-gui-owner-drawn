use glam::{Mat4, Vec3, Vec4};

/// Projects a 3D world-space point through an MVP matrix to 2D screen coordinates.
/// Returns `None` if the point is behind the camera or outside a generous clip region.
pub(crate) fn project_to_screen(
    p: Vec3,
    mvp: &[[f32; 4]; 4],
    w: f32,
    h: f32,
) -> Option<(f32, f32)> {
    let m = Mat4::from_cols_array_2d(mvp);
    let clip = m * Vec4::new(p.x, p.y, p.z, 1.0);
    if clip.w < 0.01 {
        return None;
    }
    let nx = clip.x / clip.w;
    let ny = clip.y / clip.w;
    if nx.abs() > 2.0 || ny.abs() > 2.0 {
        return None;
    }
    Some(((nx + 1.0) / 2.0 * w, (1.0 - ny) / 2.0 * h))
}

/// Converte coordenada de tela para um raio 3D (origem + direcao).
///
/// Retorna (ray_origin, ray_direction) no espaco world.
pub(crate) fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    screen_w: f32,
    screen_h: f32,
    mvp: &[[f32; 4]; 4],
) -> (Vec3, Vec3) {
    let m = Mat4::from_cols_array_2d(mvp);
    let inv = m.inverse();

    // NDC: [-1, 1]
    let nx = (screen_x / screen_w) * 2.0 - 1.0;
    let ny = 1.0 - (screen_y / screen_h) * 2.0;

    // Unproject near e far planes
    let near = inv * Vec4::new(nx, ny, 0.0, 1.0);
    let far = inv * Vec4::new(nx, ny, 1.0, 1.0);

    let near3 = Vec3::new(near.x / near.w, near.y / near.w, near.z / near.w);
    let far3 = Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);

    let dir = (far3 - near3).normalize();
    (near3, dir)
}

/// Intersecta um raio com uma esfera (para hit-test simples nos tumores).
///
/// Retorna a distancia ao ponto de interseccao, ou None se nao intersecta.
pub(crate) fn ray_sphere_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<f32> {
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let t = (-b - discriminant.sqrt()) / (2.0 * a);
    if t > 0.0 { Some(t) } else { None }
}

/// Ponto de medicao no espaco 3D.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasurePoint {
    pub world_pos: Vec3,
}

/// Calcula distancia em mm entre dois pontos no espaco normalizado.
///
/// BraTS 1mm isotropico: distancia em voxels = distancia em mm.
/// Espaco normalizado: world_pos * scale = distancia em voxels (com upsample).
/// Distancia real (mm) = distancia_world * scale / upsample.
pub(crate) fn distance_mm(a: Vec3, b: Vec3, scale: f32, upsample: f32) -> f32 {
    (a - b).length() * scale / upsample
}
