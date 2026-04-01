//! Biblioteca de widgets NeuroScan — controles UI production-grade.
//!
//! Widgets construidos sobre o sistema de primitivas (Prim2DBatch) e labels
//! existente. Zero dependencias externas. Controle total.
//!
//! Widgets disponiveis:
//! - [`Button`] — botao clicavel com hover, pressed, disabled
//! - [`TextInput`] — campo de texto com cursor, placeholder, password mode
//! - [`Checkbox`] — toggle booleano com label e checkmark
//!
//! Estilos pre-definidos em [`theme`] usando a paleta tom-sobre-tom azul.

pub mod button;
pub mod checkbox;
pub mod text_input;
pub mod theme;
pub mod types;

pub use button::Button;
pub use checkbox::Checkbox;
pub use text_input::TextInput;
pub use types::{Rect, WidgetEvent, WidgetState, WidgetStyle};
