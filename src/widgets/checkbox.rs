//! Widget Checkbox — toggle booleano com label.

use glyphon::FontSystem;

use crate::renderer::Prim2DBatch;
use crate::ui::Label;

use super::types::{Rect, WidgetEvent, WidgetState, WidgetStyle};

/// Checkbox com label.
pub struct Checkbox {
    /// Texto ao lado do checkbox.
    pub label: String,
    /// Estado checked/unchecked.
    pub checked: bool,
    /// Bounds do checkbox inteiro (box + label).
    pub bounds: Rect,
    /// Bounds apenas do box (para rendering).
    box_bounds: Rect,
    /// Estado de interacao.
    pub state: WidgetState,
    /// Estilo visual.
    pub style: WidgetStyle,
}

impl Checkbox {
    /// Cria um checkbox com label.
    pub fn new(label: &str, x: f32, y: f32, style: WidgetStyle) -> Self {
        let box_size = style.height;
        let total_w = box_size + 8.0 + 200.0; // box + gap + label estimate
        Self {
            label: label.to_string(),
            checked: false,
            bounds: Rect::new(x, y, total_w, box_size),
            box_bounds: Rect::new(x, y, box_size, box_size),
            state: WidgetState::Normal,
            style,
        }
    }

    /// Processa evento de mouse. Retorna `ValueChanged` se toggled.
    pub fn handle_mouse(
        &mut self,
        mx: f32,
        my: f32,
        _pressed: bool,
        just_released: bool,
    ) -> WidgetEvent {
        let inside = self.bounds.contains(mx, my);

        if inside && just_released {
            self.checked = !self.checked;
            self.state = WidgetState::Hovered;
            return WidgetEvent::ValueChanged;
        }

        self.state = if inside {
            WidgetState::Hovered
        } else {
            WidgetState::Normal
        };

        WidgetEvent::None
    }

    /// Renderiza o box e checkmark como primitivas 2D.
    pub fn render_prims(&self, batch: &mut Prim2DBatch, sw: f32, sh: f32) {
        let bg = self.style.bg_for_state(self.state);
        let border = self.style.border_for_state(self.state);
        let r = &self.box_bounds;

        // Box fundo
        batch.rect(r.x, r.y, r.w, r.h, bg, sw, sh);

        // Box bordas
        batch.rect(r.x, r.y, r.w, 1.0, border, sw, sh);
        batch.rect(r.x, r.bottom() - 1.0, r.w, 1.0, border, sw, sh);
        batch.rect(r.x, r.y, 1.0, r.h, border, sw, sh);
        batch.rect(r.right() - 1.0, r.y, 1.0, r.h, border, sw, sh);

        // Checkmark (quando checked): duas linhas formando V
        if self.checked {
            let cx = r.x + r.w / 2.0;
            let cy = r.y + r.h / 2.0;
            let s = r.w * 0.3; // tamanho do check
            let check_color = [0.30, 0.70, 1.0, 0.95]; // azul brilhante

            // Perna esquerda do V (de cima-esquerda para centro-baixo)
            batch.line(
                cx - s,
                cy - s * 0.2,
                cx - s * 0.2,
                cy + s,
                check_color,
                2.0,
                sw,
                sh,
            );
            // Perna direita do V (de centro-baixo para cima-direita)
            batch.line(
                cx - s * 0.2,
                cy + s,
                cx + s,
                cy - s * 0.8,
                check_color,
                2.0,
                sw,
                sh,
            );

            // Fundo azul sutil quando checked
            batch.rect(
                r.x + 2.0,
                r.y + 2.0,
                r.w - 4.0,
                r.h - 4.0,
                [0.15, 0.40, 0.70, 0.30],
                sw,
                sh,
            );
        }
    }

    /// Renderiza o label ao lado do checkbox.
    pub fn render_label(&self, fs: &mut FontSystem) -> Label {
        Label::new(
            fs,
            &self.label,
            self.style.text_size,
            self.style.text_color,
            self.box_bounds.right() + 8.0,
            self.bounds.y + (self.bounds.h - self.style.text_size) / 2.0,
        )
    }
}
