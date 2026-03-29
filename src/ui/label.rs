use glyphon::cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, Weight};
use glyphon::{TextArea, TextBounds};

use super::Color;

/// Componente de texto estatico ou dinamico.
///
/// Equivale ao `TLabel` do Delphi VCL: encapsula conteudo, fonte, cor e
/// posicao. Sabe se preparar para renderizacao sem que o caller precise
/// conhecer os internos do glyphon.
///
/// # Exemplo
/// ```rust
/// let mut title = Label::new_bold(fs, "Andar 5", 52.0, Color::WHITE, 0.0, 0.0);
/// title.x = (screen_w - title.measured_width()) / 2.0; // centralizar
/// ```
pub struct Label {
    /// Posicao horizontal em pixels logicos a partir da borda esquerda.
    pub x: f32,
    /// Posicao vertical em pixels logicos a partir do topo.
    pub y: f32,
    /// Cor do texto.
    pub color: Color,
    #[allow(dead_code)] // lido em set_text() — API publica nao exercida ainda neste crate
    bold: bool,
    font_size: f32,
    buffer: Buffer,
}

impl Label {
    /// Cria um Label em Inter Regular.
    pub fn new(
        font_system: &mut FontSystem,
        text: &str,
        font_size: f32,
        color: Color,
        x: f32,
        y: f32,
    ) -> Self {
        Self::build(font_system, text, font_size, color, x, y, false)
    }

    /// Cria um Label em Inter Bold.
    pub fn new_bold(
        font_system: &mut FontSystem,
        text: &str,
        font_size: f32,
        color: Color,
        x: f32,
        y: f32,
    ) -> Self {
        Self::build(font_system, text, font_size, color, x, y, true)
    }

    fn build(
        font_system: &mut FontSystem,
        text: &str,
        font_size: f32,
        color: Color,
        x: f32,
        y: f32,
        bold: bool,
    ) -> Self {
        let line_height = font_size * 1.25;
        let mut buffer = Buffer::new(font_system, Metrics::new(font_size, line_height));
        // Sem wrap: o buffer cresce horizontalmente com o texto.
        // A posicao e responsabilidade do caller (layout da tela).
        buffer.set_size(font_system, None, None);
        buffer.set_text(font_system, text, &Self::attrs(bold), Shaping::Advanced);
        buffer.shape_until_scroll(font_system, false);
        Self {
            x,
            y,
            color,
            bold,
            font_size,
            buffer,
        }
    }

    /// Atualiza o texto em runtime. Acionar quando o conteudo mudar (ex: numero do andar).
    #[allow(dead_code)]
    pub fn set_text(&mut self, font_system: &mut FontSystem, text: &str) {
        self.buffer.set_text(
            font_system,
            text,
            &Self::attrs(self.bold),
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(font_system, false);
    }

    /// Largura medida em pixels apos shaping. Usar para centralizar horizontalmente:
    /// `label.x = (screen_w - label.measured_width()) / 2.0`
    pub fn measured_width(&self) -> f32 {
        self.buffer
            .layout_runs()
            .map(|r| r.line_w)
            .fold(0.0_f32, f32::max)
    }

    /// Altura da linha em pixels. Usar para empilhar labels verticalmente:
    /// `next_label.y = prev_label.y + prev_label.line_height()`
    pub fn line_height(&self) -> f32 {
        self.font_size * 1.25
    }

    /// Produz o TextArea para `TextRenderer::prepare()`. Uso exclusivo do renderer.
    pub(crate) fn as_text_area(&self, screen_w: u32, screen_h: u32) -> TextArea<'_> {
        TextArea {
            buffer: &self.buffer,
            left: self.x,
            top: self.y,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: screen_w as i32,
                bottom: screen_h as i32,
            },
            default_color: self.color.to_glyphon(),
            custom_glyphs: &[],
        }
    }

    fn attrs(bold: bool) -> Attrs<'static> {
        let base = Attrs::new().family(Family::Name("Inter"));
        if bold {
            base.weight(Weight::BOLD)
        } else {
            base
        }
    }
}
