mod labels;
mod primitives;

use crate::ui::Color;

// ---------------------------------------------------------------------------
// Paleta tom-sobre-tom azul (blue monochrome, do escuro ao claro)
// ---------------------------------------------------------------------------

// Fundos (usados como [f32;4] RGBA linear nos primitivos)
/// Fundo mais profundo — near-black com tom azul
pub(crate) const BG_DEEP: [f32; 4] = [0.035, 0.055, 0.085, 1.0];
/// Fundo de superficie (boxes, paineis)
pub(crate) const BG_SURFACE: [f32; 4] = [0.050, 0.075, 0.115, 0.92];
/// Borda/separador sutil
pub(crate) const BG_BORDER: [f32; 4] = [0.10, 0.16, 0.26, 0.40];
/// Glow/accent azul (barra de progresso, highlights)
pub(crate) const ACCENT: [f32; 4] = [0.20, 0.55, 0.88, 0.90];
/// Glow suave (halos, pulsos)
pub(crate) const ACCENT_GLOW: [f32; 4] = [0.20, 0.55, 0.88, 0.12];

// Texto (Color = sRGB u8)

/// Titulo principal — azul quase branco
pub(crate) fn col_header() -> Color {
    Color::rgb(200, 222, 245)
}

/// Texto secundario/valores — azul claro
pub(crate) fn col_value() -> Color {
    Color::rgb(160, 192, 220)
}

/// Texto dim (timestamps, footers) — azul-acinzentado
pub(crate) fn col_dim() -> Color {
    Color::rgb(80, 110, 148)
}

/// Titulos de secao — azul medio
pub(crate) fn col_section() -> Color {
    Color::rgb(65, 95, 130)
}

/// Subtitulos — azul intermediario
pub(crate) fn col_subtitle() -> Color {
    Color::rgb(100, 145, 195)
}

/// Separador (RGBA linear)
#[allow(dead_code)]
pub(crate) fn col_sep() -> [f32; 4] {
    BG_BORDER
}

/// Converte [f32;3] linear para Color sRGB.
pub(crate) fn rgb_f(c: [f32; 3]) -> Color {
    Color::rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}
