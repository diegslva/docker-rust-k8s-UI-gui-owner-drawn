//! Estilos pre-definidos para widgets — paleta tom-sobre-tom azul NeuroScan.

use crate::ui::Color;

use super::types::WidgetStyle;

/// Botao primario — acao principal (azul accent).
pub fn style_primary() -> WidgetStyle {
    WidgetStyle {
        bg: [0.15, 0.40, 0.70, 0.95],
        bg_hover: [0.18, 0.48, 0.80, 0.98],
        bg_pressed: [0.12, 0.35, 0.60, 1.0],
        bg_disabled: [0.08, 0.15, 0.28, 0.50],
        border: [0.20, 0.50, 0.85, 0.60],
        border_focus: [0.30, 0.65, 1.0, 0.90],
        text_color: Color::rgb(230, 240, 255),
        text_placeholder: Color::rgb(120, 150, 180),
        text_size: 13.0,
        padding_h: 20.0,
        padding_v: 8.0,
        height: 36.0,
    }
}

/// Botao secundario — acao alternativa (outline).
pub fn style_secondary() -> WidgetStyle {
    WidgetStyle {
        bg: [0.04, 0.06, 0.10, 0.60],
        bg_hover: [0.06, 0.10, 0.18, 0.80],
        bg_pressed: [0.04, 0.08, 0.14, 0.90],
        bg_disabled: [0.04, 0.06, 0.10, 0.30],
        border: [0.15, 0.30, 0.55, 0.50],
        border_focus: [0.25, 0.55, 0.90, 0.80],
        text_color: Color::rgb(160, 195, 230),
        text_placeholder: Color::rgb(80, 110, 148),
        text_size: 13.0,
        padding_h: 20.0,
        padding_v: 8.0,
        height: 36.0,
    }
}

/// Campo de texto (TextInput).
pub fn style_input() -> WidgetStyle {
    WidgetStyle {
        bg: [0.03, 0.05, 0.09, 0.90],
        bg_hover: [0.04, 0.07, 0.12, 0.95],
        bg_pressed: [0.03, 0.05, 0.09, 0.90],
        bg_disabled: [0.03, 0.04, 0.07, 0.50],
        border: [0.10, 0.18, 0.32, 0.60],
        border_focus: [0.20, 0.55, 0.88, 0.90],
        text_color: Color::rgb(200, 220, 240),
        text_placeholder: Color::rgb(70, 95, 130),
        text_size: 13.0,
        padding_h: 12.0,
        padding_v: 8.0,
        height: 36.0,
    }
}

/// Checkbox.
pub fn style_checkbox() -> WidgetStyle {
    WidgetStyle {
        bg: [0.04, 0.07, 0.12, 0.80],
        bg_hover: [0.06, 0.10, 0.18, 0.90],
        bg_pressed: [0.08, 0.14, 0.24, 0.95],
        bg_disabled: [0.04, 0.06, 0.10, 0.40],
        border: [0.12, 0.22, 0.40, 0.60],
        border_focus: [0.20, 0.55, 0.88, 0.90],
        text_color: Color::rgb(180, 200, 225),
        text_placeholder: Color::rgb(80, 110, 148),
        text_size: 12.0,
        padding_h: 8.0,
        padding_v: 4.0,
        height: 24.0,
    }
}

/// Botao de acao destrutiva (vermelho).
#[allow(dead_code)]
pub fn style_danger() -> WidgetStyle {
    WidgetStyle {
        bg: [0.60, 0.12, 0.10, 0.90],
        bg_hover: [0.75, 0.15, 0.12, 0.95],
        bg_pressed: [0.50, 0.10, 0.08, 1.0],
        bg_disabled: [0.25, 0.08, 0.07, 0.50],
        border: [0.80, 0.20, 0.15, 0.60],
        border_focus: [0.95, 0.30, 0.20, 0.90],
        text_color: Color::rgb(255, 230, 225),
        text_placeholder: Color::rgb(180, 120, 115),
        text_size: 13.0,
        padding_h: 20.0,
        padding_v: 8.0,
        height: 36.0,
    }
}
