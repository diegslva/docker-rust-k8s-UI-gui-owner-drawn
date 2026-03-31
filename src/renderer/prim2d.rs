// ---------------------------------------------------------------------------
// Primitivas 2D — geometria colorida em NDC (callout lines, boxes, separadores)
// ---------------------------------------------------------------------------

/// Vértice de primitiva 2D: posição NDC + cor RGBA.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VertexPrim {
    pub pos: [f32; 2],
    pub col: [f32; 4],
}

/// Batch de geometria 2D colorida construido CPU-side a cada frame.
///
/// Coordenadas em pixels de tela; conversao para NDC e interna.
/// Suporta retangulos preenchidos e linhas como quads finos.
pub struct Prim2DBatch {
    pub(crate) verts: Vec<VertexPrim>,
    pub(crate) indices: Vec<u32>,
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

    pub(crate) fn px_to_ndc(px: f32, py: f32, sw: f32, sh: f32) -> [f32; 2] {
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

impl Default for Prim2DBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Propriedades de renderizacao por mesh.
pub struct MeshEntry<'a> {
    pub mesh: &'a crate::mesh::Mesh,
    pub tint: [f32; 3],
    /// alpha: 1.0 = opaco, <1.0 = semi-transparente (alpha blend).
    pub alpha: f32,
    /// roughness: 0.0 = espelho, 1.0 = fosco. Tumores ~0.3, cerebro ~0.7.
    pub roughness: f32,
    /// sss_strength: intensidade de subsurface scattering (0.0 = off).
    pub sss_strength: f32,
    /// use_texture: > 0.5 = sample diffuse texture, <= 0.5 = use tint.
    pub use_texture: f32,
}
