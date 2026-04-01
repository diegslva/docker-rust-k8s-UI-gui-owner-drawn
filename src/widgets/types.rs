//! Tipos base da biblioteca de widgets NeuroScan.
//!
//! Fundacao para todos os controles UI: Button, TextInput, Checkbox, etc.

use crate::ui::Color;

/// Retangulo de bounds — hit testing e layout.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Testa se o ponto (mx, my) esta dentro do retangulo.
    pub fn contains(&self, mx: f32, my: f32) -> bool {
        mx >= self.x && mx < self.x + self.w && my >= self.y && my < self.y + self.h
    }

    /// Centro do retangulo.
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Borda direita.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Borda inferior.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
}

/// Estado de interacao de um widget.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WidgetState {
    /// Nenhuma interacao.
    Normal,
    /// Mouse sobre o widget.
    Hovered,
    /// Mouse pressionado sobre o widget.
    Pressed,
    /// Widget desabilitado (nao responde a input).
    Disabled,
    /// Widget com foco de teclado (para TextInput).
    Focused,
}

/// Resultado de um evento processado por um widget.
#[derive(Clone, Debug, PartialEq)]
pub enum WidgetEvent {
    /// Nenhuma acao.
    None,
    /// Widget foi clicado (Button).
    Clicked,
    /// Valor booleano mudou (Checkbox).
    ValueChanged,
    /// Texto mudou (TextInput).
    TextChanged(String),
}

/// Estilo visual de um widget — cores, tamanhos, padding.
///
/// Cada widget pode ter estilo custom ou usar os pre-definidos de `theme.rs`.
#[derive(Clone, Debug)]
pub struct WidgetStyle {
    /// Fundo no estado normal.
    pub bg: [f32; 4],
    /// Fundo no hover.
    pub bg_hover: [f32; 4],
    /// Fundo quando pressionado.
    pub bg_pressed: [f32; 4],
    /// Fundo quando desabilitado.
    pub bg_disabled: [f32; 4],
    /// Cor da borda.
    pub border: [f32; 4],
    /// Cor da borda quando focado.
    pub border_focus: [f32; 4],
    /// Cor do texto.
    pub text_color: Color,
    /// Cor do texto placeholder (dim).
    pub text_placeholder: Color,
    /// Tamanho da fonte.
    pub text_size: f32,
    /// Padding interno horizontal.
    pub padding_h: f32,
    /// Padding interno vertical.
    pub padding_v: f32,
    /// Altura total do widget.
    pub height: f32,
}

impl WidgetStyle {
    /// Retorna a cor de fundo para o estado atual.
    pub fn bg_for_state(&self, state: WidgetState) -> [f32; 4] {
        match state {
            WidgetState::Normal | WidgetState::Focused => self.bg,
            WidgetState::Hovered => self.bg_hover,
            WidgetState::Pressed => self.bg_pressed,
            WidgetState::Disabled => self.bg_disabled,
        }
    }

    /// Retorna a cor da borda para o estado atual.
    pub fn border_for_state(&self, state: WidgetState) -> [f32; 4] {
        match state {
            WidgetState::Focused => self.border_focus,
            _ => self.border,
        }
    }
}
