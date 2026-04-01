//! Tela de login do NeuroScan — autenticacao antes do acesso ao software.
//!
//! Widgets: 2 TextInput (usuario, senha), 1 Button (Entrar), 1 Checkbox (Lembrar-me).
//! Aparece apos o splash e antes da home screen.

use glyphon::FontSystem;

use crate::renderer::Prim2DBatch;
use crate::ui::{Color, Label};
use crate::widgets::{Button, Checkbox, TextInput, WidgetEvent};

use super::ui_builder::{BG_DEEP, BG_SURFACE, col_dim, col_header, col_subtitle};

/// Estado da tela de login.
pub(crate) struct LoginScreen {
    pub username_input: TextInput,
    pub password_input: TextInput,
    pub login_button: Button,
    pub remember_checkbox: Checkbox,
    /// Mensagem de erro (credenciais invalidas, etc.).
    pub error_message: Option<String>,
    /// Se o login foi bem-sucedido.
    pub authenticated: bool,
}

impl LoginScreen {
    /// Cria a tela de login com widgets centralizados.
    pub fn new(screen_w: f32, screen_h: f32) -> Self {
        let form_w = 320.0_f32;
        let cx = (screen_w - form_w) / 2.0;
        let mut y = screen_h * 0.38;

        let input_style = crate::widgets::theme::style_input();
        let button_style = crate::widgets::theme::style_primary();
        let checkbox_style = crate::widgets::theme::style_checkbox();

        let username_input = TextInput::new("Usuario", cx, y, form_w, input_style.clone());
        y += 48.0;

        let mut password_input = TextInput::new_password("Senha", cx, y, form_w, input_style);
        y += 56.0;

        let login_button = Button::new("Entrar", cx, y, form_w, button_style);
        y += 48.0;

        let remember_checkbox = Checkbox::new("Lembrar-me", cx, y, checkbox_style);

        Self {
            username_input,
            password_input,
            login_button,
            remember_checkbox,
            error_message: None,
            authenticated: false,
        }
    }

    /// Reposiciona widgets quando a janela e redimensionada.
    pub fn resize(&mut self, screen_w: f32, screen_h: f32) {
        let form_w = 320.0_f32;
        let cx = (screen_w - form_w) / 2.0;
        let mut y = screen_h * 0.38;

        self.username_input.bounds.x = cx;
        self.username_input.bounds.y = y;
        self.username_input.bounds.w = form_w;
        y += 48.0;

        self.password_input.bounds.x = cx;
        self.password_input.bounds.y = y;
        self.password_input.bounds.w = form_w;
        y += 56.0;

        self.login_button.bounds.x = cx;
        self.login_button.bounds.y = y;
        self.login_button.bounds.w = form_w;
        y += 48.0;

        self.remember_checkbox.bounds.x = cx;
        self.remember_checkbox.bounds.y = y;
    }

    /// Processa evento de mouse em todos os widgets.
    /// Retorna true se o login foi tentado.
    pub fn handle_mouse(&mut self, mx: f32, my: f32, pressed: bool, just_released: bool) -> bool {
        self.username_input
            .handle_mouse(mx, my, pressed, just_released);
        self.password_input
            .handle_mouse(mx, my, pressed, just_released);
        self.remember_checkbox
            .handle_mouse(mx, my, pressed, just_released);

        let btn_event = self
            .login_button
            .handle_mouse(mx, my, pressed, just_released);

        if btn_event == WidgetEvent::Clicked {
            return self.attempt_login();
        }
        false
    }

    /// Processa tecla especial (delegada ao input focado).
    pub fn handle_key(&mut self, key: &winit::keyboard::Key) -> bool {
        // Enter = tentar login
        if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) = key {
            return self.attempt_login();
        }

        // Tab = alternar foco entre campos
        if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab) = key {
            if self.username_input.state == crate::widgets::WidgetState::Focused {
                self.username_input.state = crate::widgets::WidgetState::Normal;
                self.password_input.state = crate::widgets::WidgetState::Focused;
                self.password_input.cursor_pos = self.password_input.text.len();
            } else {
                self.password_input.state = crate::widgets::WidgetState::Normal;
                self.username_input.state = crate::widgets::WidgetState::Focused;
                self.username_input.cursor_pos = self.username_input.text.len();
            }
            return false;
        }

        self.username_input.handle_key(key);
        self.password_input.handle_key(key);
        false
    }

    /// Processa caractere digitado.
    pub fn handle_char(&mut self, ch: char) {
        self.username_input.handle_char(ch);
        self.password_input.handle_char(ch);
    }

    /// Atualiza animacoes (cursor piscante).
    pub fn update(&mut self, dt: f32) {
        self.username_input.update(dt);
        self.password_input.update(dt);
    }

    /// Tenta autenticar. Por enquanto, aceita qualquer credencial nao-vazia.
    /// V2: verificar contra banco local (SQLite + bcrypt).
    fn attempt_login(&mut self) -> bool {
        let user = self.username_input.text.trim();
        let pass = self.password_input.text.trim();

        if user.is_empty() || pass.is_empty() {
            self.error_message = Some("Preencha usuario e senha".to_string());
            return false;
        }

        // V1: aceitar qualquer credencial nao-vazia
        // V2: verificar hash bcrypt no SQLite
        self.authenticated = true;
        self.error_message = None;
        tracing::info!(user = %user, "login bem-sucedido");
        true
    }

    /// Renderiza primitivas (fundos dos widgets + fundo da tela).
    pub fn render_prims(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();

        // Fundo escuro
        b.rect(0.0, 0.0, w, h, BG_DEEP, w, h);

        // Card central (fundo sutil)
        let form_w = 360.0;
        let form_h = 320.0;
        let fx = (w - form_w) / 2.0;
        let fy = h * 0.22;
        b.rect(fx, fy, form_w, form_h, BG_SURFACE, w, h);
        // Borda superior azul
        b.rect(fx, fy, form_w, 2.0, [0.20, 0.55, 0.88, 0.70], w, h);

        // Widgets
        self.username_input.render_prims(&mut b, w, h);
        self.password_input.render_prims(&mut b, w, h);
        self.login_button.render_prims(&mut b, w, h);
        self.remember_checkbox.render_prims(&mut b, w, h);

        b
    }

    /// Renderiza labels (textos dos widgets + titulo + erro).
    pub fn render_labels(&self, fs: &mut FontSystem, w: f32, h: f32) -> Vec<Label> {
        let mut labels = Vec::new();

        // Titulo
        let mut title = Label::new_bold(fs, "NeuroScan AI", 32.0, col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.14;
        labels.push(title);

        // Subtitulo
        let mut sub = Label::new(fs, "Acesso ao sistema", 13.0, col_subtitle(), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = h * 0.14 + 42.0;
        labels.push(sub);

        // Labels dos widgets
        labels.push(self.username_input.render_label(fs));
        labels.push(self.password_input.render_label(fs));
        labels.push(self.login_button.render_label(fs));
        labels.push(self.remember_checkbox.render_label(fs));

        // Mensagem de erro
        if let Some(err) = &self.error_message {
            let mut err_lbl = Label::new_bold(fs, err, 11.0, Color::rgb(240, 120, 100), 0.0, 0.0);
            err_lbl.x = (w - err_lbl.measured_width()) / 2.0;
            err_lbl.y = h * 0.38 + 200.0;
            labels.push(err_lbl);
        }

        // Versao no rodape
        let ver = format!("NeuroScan AI  v{}", env!("CARGO_PKG_VERSION"));
        let mut vl = Label::new(fs, &ver, 9.0, col_dim(), 0.0, 0.0);
        vl.x = (w - vl.measured_width()) / 2.0;
        vl.y = h - 28.0;
        labels.push(vl);

        labels
    }
}
