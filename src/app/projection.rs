use glam::{Mat4, Vec4};

/// Projects a 3D world-space point through an MVP matrix to 2D screen coordinates.
/// Returns `None` if the point is behind the camera or outside a generous clip region.
pub(crate) fn project_to_screen(
    p: glam::Vec3,
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
