//! Widget Button — botao clicavel com hover, pressed e disabled states.

use glyphon::FontSystem;

use crate::renderer::Prim2DBatch;
use crate::ui::{Color, Label};

use super::types::{Rect, WidgetEvent, WidgetState, WidgetStyle};

/// Botao clicavel.
pub struct Button {
    /// Texto exibido no botao.
    pub label: String,
    /// Bounds para rendering e hit testing.
    pub bounds: Rect,
    /// Estado atual de interacao.
    pub state: WidgetState,
    /// Estilo visual.
    pub style: WidgetStyle,
    /// Se o botao esta habilitado.
    pub enabled: bool,
}

impl Button {
    /// Cria um botao com posicao e largura especificas.
    /// A altura vem do estilo.
    pub fn new(label: &str, x: f32, y: f32, w: f32, style: WidgetStyle) -> Self {
        let h = style.height;
        Self {
            label: label.to_string(),
            bounds: Rect::new(x, y, w, h),
            state: WidgetState::Normal,
            style,
            enabled: true,
        }
    }

    /// Processa evento de mouse. Retorna `WidgetEvent::Clicked` se clicado.
    ///
    /// `pressed`: true se botao esquerdo esta pressionado neste frame.
    /// `just_released`: true se o botao foi solto neste frame (click completo).
    pub fn handle_mouse(
        &mut self,
        mx: f32,
        my: f32,
        pressed: bool,
        just_released: bool,
    ) -> WidgetEvent {
        if !self.enabled {
            self.state = WidgetState::Disabled;
            return WidgetEvent::None;
        }

        let inside = self.bounds.contains(mx, my);

        if inside && just_released {
            self.state = WidgetState::Hovered;
            return WidgetEvent::Clicked;
        }

        self.state = if inside && pressed {
            WidgetState::Pressed
        } else if inside {
            WidgetState::Hovered
        } else {
            WidgetState::Normal
        };

        WidgetEvent::None
    }

    /// Renderiza o fundo e bordas do botao como primitivas 2D.
    pub fn render_prims(&self, batch: &mut Prim2DBatch, sw: f32, sh: f32) {
        let state = if self.enabled {
            self.state
        } else {
            WidgetState::Disabled
        };
        let bg = self.style.bg_for_state(state);
        let border = self.style.border_for_state(state);
        let r = &self.bounds;

        // Fundo
        batch.rect(r.x, r.y, r.w, r.h, bg, sw, sh);

        // Bordas (1px cada lado)
        batch.rect(r.x, r.y, r.w, 1.0, border, sw, sh); // top
        batch.rect(r.x, r.bottom() - 1.0, r.w, 1.0, border, sw, sh); // bottom
        batch.rect(r.x, r.y, 1.0, r.h, border, sw, sh); // left
        batch.rect(r.right() - 1.0, r.y, 1.0, r.h, border, sw, sh); // right

        // Highlight sutil no topo (efeito 3D)
        if state == WidgetState::Hovered || state == WidgetState::Normal {
            batch.rect(
                r.x + 1.0,
                r.y + 1.0,
                r.w - 2.0,
                1.0,
                [1.0, 1.0, 1.0, 0.05],
                sw,
                sh,
            );
        }
    }

    /// Renderiza o texto do botao como Label (centralizado).
    pub fn render_label(&self, fs: &mut FontSystem) -> Label {
        let color = if self.enabled {
            self.style.text_color
        } else {
            Color::rgb(80, 95, 115)
        };

        let mut lbl = Label::new_bold(fs, &self.label, self.style.text_size, color, 0.0, 0.0);
        // Centralizar horizontalmente
        lbl.x = self.bounds.x + (self.bounds.w - lbl.measured_width()) / 2.0;
        // Centralizar verticalmente
        lbl.y = self.bounds.y + (self.bounds.h - self.style.text_size) / 2.0;
        lbl
    }
}
