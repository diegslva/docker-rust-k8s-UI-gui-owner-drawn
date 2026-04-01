//! Widget TextInput — campo de texto editavel com cursor, placeholder e password mode.

use glyphon::FontSystem;

use crate::renderer::Prim2DBatch;
use crate::ui::{Color, Label};

use super::types::{Rect, WidgetEvent, WidgetState, WidgetStyle};

/// Campo de texto editavel.
pub struct TextInput {
    /// Texto atual.
    pub text: String,
    /// Texto placeholder (exibido quando vazio).
    pub placeholder: String,
    /// Bounds para rendering e hit testing.
    pub bounds: Rect,
    /// Estado de interacao.
    pub state: WidgetState,
    /// Estilo visual.
    pub style: WidgetStyle,
    /// Posicao do cursor (indice de caractere).
    pub cursor_pos: usize,
    /// Modo senha (mostra asteriscos).
    pub is_password: bool,
    /// Comprimento maximo (0 = sem limite).
    pub max_length: usize,
    /// Timer para cursor piscante.
    pub cursor_blink_t: f32,
}

impl TextInput {
    /// Cria um campo de texto.
    pub fn new(placeholder: &str, x: f32, y: f32, w: f32, style: WidgetStyle) -> Self {
        let h = style.height;
        Self {
            text: String::new(),
            placeholder: placeholder.to_string(),
            bounds: Rect::new(x, y, w, h),
            state: WidgetState::Normal,
            style,
            cursor_pos: 0,
            is_password: false,
            max_length: 0,
            cursor_blink_t: 0.0,
        }
    }

    /// Cria um campo de senha.
    pub fn new_password(placeholder: &str, x: f32, y: f32, w: f32, style: WidgetStyle) -> Self {
        let mut input = Self::new(placeholder, x, y, w, style);
        input.is_password = true;
        input
    }

    /// Processa clique do mouse — foca/desfoca o campo.
    pub fn handle_mouse(
        &mut self,
        mx: f32,
        my: f32,
        pressed: bool,
        just_released: bool,
    ) -> WidgetEvent {
        let inside = self.bounds.contains(mx, my);

        if inside && just_released {
            self.state = WidgetState::Focused;
            self.cursor_blink_t = 0.0;
            // Posicionar cursor no final do texto ao clicar
            self.cursor_pos = self.text.len();
            return WidgetEvent::None;
        }

        if !inside && just_released && self.state == WidgetState::Focused {
            self.state = WidgetState::Normal;
            return WidgetEvent::None;
        }

        if self.state != WidgetState::Focused {
            self.state = if inside && pressed {
                WidgetState::Pressed
            } else if inside {
                WidgetState::Hovered
            } else {
                WidgetState::Normal
            };
        }

        WidgetEvent::None
    }

    /// Processa tecla especial (Backspace, Delete, Home, End, etc.).
    pub fn handle_key(&mut self, key: &winit::keyboard::Key) -> WidgetEvent {
        if self.state != WidgetState::Focused {
            return WidgetEvent::None;
        }

        use winit::keyboard::NamedKey;
        match key {
            winit::keyboard::Key::Named(NamedKey::Backspace) => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.text.remove(self.cursor_pos);
                    self.cursor_blink_t = 0.0;
                    return WidgetEvent::TextChanged(self.text.clone());
                }
            }
            winit::keyboard::Key::Named(NamedKey::Delete) => {
                if self.cursor_pos < self.text.len() {
                    self.text.remove(self.cursor_pos);
                    self.cursor_blink_t = 0.0;
                    return WidgetEvent::TextChanged(self.text.clone());
                }
            }
            winit::keyboard::Key::Named(NamedKey::Home) => {
                self.cursor_pos = 0;
                self.cursor_blink_t = 0.0;
            }
            winit::keyboard::Key::Named(NamedKey::End) => {
                self.cursor_pos = self.text.len();
                self.cursor_blink_t = 0.0;
            }
            winit::keyboard::Key::Named(NamedKey::ArrowLeft) => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.cursor_blink_t = 0.0;
                }
            }
            winit::keyboard::Key::Named(NamedKey::ArrowRight) => {
                if self.cursor_pos < self.text.len() {
                    self.cursor_pos += 1;
                    self.cursor_blink_t = 0.0;
                }
            }
            _ => {}
        }

        WidgetEvent::None
    }

    /// Processa caractere digitado.
    pub fn handle_char(&mut self, ch: char) -> WidgetEvent {
        if self.state != WidgetState::Focused {
            return WidgetEvent::None;
        }

        // Ignorar caracteres de controle
        if ch.is_control() {
            return WidgetEvent::None;
        }

        // Verificar limite de comprimento
        if self.max_length > 0 && self.text.len() >= self.max_length {
            return WidgetEvent::None;
        }

        self.text.insert(self.cursor_pos, ch);
        self.cursor_pos += 1;
        self.cursor_blink_t = 0.0;

        WidgetEvent::TextChanged(self.text.clone())
    }

    /// Atualiza timer do cursor piscante.
    pub fn update(&mut self, dt: f32) {
        if self.state == WidgetState::Focused {
            self.cursor_blink_t += dt;
        }
    }

    /// Texto exibido (asteriscos se password).
    fn display_text(&self) -> String {
        if self.is_password {
            "\u{2022}".repeat(self.text.len()) // bullet character
        } else {
            self.text.clone()
        }
    }

    /// Renderiza o fundo, bordas e cursor como primitivas 2D.
    pub fn render_prims(&self, batch: &mut Prim2DBatch, sw: f32, sh: f32) {
        let bg = self.style.bg_for_state(self.state);
        let border = self.style.border_for_state(self.state);
        let r = &self.bounds;

        // Fundo
        batch.rect(r.x, r.y, r.w, r.h, bg, sw, sh);

        // Bordas
        batch.rect(r.x, r.y, r.w, 1.0, border, sw, sh);
        batch.rect(r.x, r.bottom() - 1.0, r.w, 1.0, border, sw, sh);
        batch.rect(r.x, r.y, 1.0, r.h, border, sw, sh);
        batch.rect(r.right() - 1.0, r.y, 1.0, r.h, border, sw, sh);

        // Cursor piscante (visivel 0.5s, invisivel 0.5s)
        if self.state == WidgetState::Focused {
            let blink_on = (self.cursor_blink_t * 2.0) as u32 % 2 == 0;
            if blink_on {
                // Estimar posicao X do cursor baseado em caracteres
                let char_width = self.style.text_size * 0.6; // estimativa
                let cursor_x = r.x + self.style.padding_h + self.cursor_pos as f32 * char_width;
                let cursor_y = r.y + self.style.padding_v;
                let cursor_h = r.h - self.style.padding_v * 2.0;
                batch.rect(
                    cursor_x,
                    cursor_y,
                    1.5,
                    cursor_h,
                    [0.70, 0.85, 1.0, 0.90],
                    sw,
                    sh,
                );
            }
        }
    }

    /// Renderiza o texto (ou placeholder) como Label.
    pub fn render_label(&self, fs: &mut FontSystem) -> Label {
        let display = self.display_text();
        let (text, color) = if display.is_empty() {
            (&self.placeholder, self.style.text_placeholder)
        } else {
            (&display, self.style.text_color)
        };

        Label::new(
            fs,
            text,
            self.style.text_size,
            color,
            self.bounds.x + self.style.padding_h,
            self.bounds.y + (self.bounds.h - self.style.text_size) / 2.0,
        )
    }
}
