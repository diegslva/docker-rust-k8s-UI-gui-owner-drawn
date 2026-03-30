mod labels;
mod primitives;

use crate::ui::Color;

// ---------------------------------------------------------------------------
// Color palette helpers
// ---------------------------------------------------------------------------

pub(crate) fn col_header() -> Color {
    Color::rgb(226, 232, 240)
}

pub(crate) fn col_dim() -> Color {
    Color::rgb(94, 111, 133)
}

pub(crate) fn col_value() -> Color {
    Color::rgb(203, 213, 225)
}

pub(crate) fn col_section() -> Color {
    Color::rgb(71, 85, 105)
}

#[allow(dead_code)]
pub(crate) fn col_sep() -> [f32; 4] {
    [0.12, 0.17, 0.26, 0.70]
}

pub(crate) fn rgb_f(c: [f32; 3]) -> Color {
    Color::rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}
