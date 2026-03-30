use neuroscan_core::{MENU_BAR_H, MENU_DROP_W, MENU_ITEM_H, MENU_SEP_H, MENU_TOP_XS};
use winit::dpi::PhysicalSize;

use crate::ui::{Color, Label};

use super::state::App;

impl App {
    /// Retorna as entradas do menu `menu_id` como (texto, shortcut_hint, is_separator).
    pub(crate) fn build_menu_entries(&self, menu_id: i32) -> Vec<(String, String, bool)> {
        neuroscan_core::menu_entries(menu_id, self.current_case, env!("CARGO_PKG_VERSION"))
    }

    /// Altura total do dropdown para o menu `menu_id`.
    pub(crate) fn dropdown_height(&self, menu_id: i32) -> f32 {
        neuroscan_core::dropdown_height(menu_id, self.current_case)
    }

    /// Constrói `labels_menu` para o dropdown do menu atualmente aberto.
    /// Chamado quando `menu_open` muda.
    pub(crate) fn rebuild_menu_labels(&mut self, size: PhysicalSize<u32>) {
        if self.menu_open < 0 {
            self.labels_menu = Vec::new();
            return;
        }
        let menu_id = self.menu_open;
        let entries = self.build_menu_entries(menu_id);
        let top_x = MENU_TOP_XS[menu_id as usize];
        let text_x = top_x + 12.0;
        let right_x = top_x + MENU_DROP_W - 14.0;
        let w = size.width as f32;
        let h = size.height as f32;

        let Some(gpu) = &mut self.gpu else { return };
        let fs = gpu.font_system_mut();
        let col_item = Color::rgb(188, 200, 218);
        let col_sc = Color::rgb(100, 116, 139);
        let col_check = Color::rgb(100, 200, 140);

        let mut labels = Vec::new();
        let mut y = MENU_BAR_H + 2.0;

        for (label, shortcut, is_sep) in &entries {
            if *is_sep {
                y += MENU_SEP_H;
                continue;
            }
            let col = if label.starts_with('\u{2713}') {
                col_check
            } else {
                col_item
            };
            labels.push(Label::new(fs, label, 11.0, col, text_x, y + 4.0));
            if !shortcut.is_empty() {
                let mut sc = Label::new(fs, shortcut, 11.0, col_sc, 0.0, y + 4.0);
                sc.x = right_x - sc.measured_width();
                labels.push(sc);
            }
            y += MENU_ITEM_H;
        }
        let _ = (w, h);
        self.labels_menu = labels;
    }

    /// Atualiza `.x`/`.y` dos labels SNFH sem re-shaping — chamado a cada frame durante animação.
    pub(crate) fn update_snfh_label_positions(&mut self, size: PhysicalSize<u32>) {
        if self.labels_snfh.is_empty() {
            return;
        }
        let ease = {
            let t = self.snfh_anim_t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let w = size.width as f32;
        let h = size.height as f32;
        let y_ct = h * 0.14;
        let box_w = (w * 0.175).clamp(190.0, 240.0);
        let box_h = 92.0_f32;
        let pad = 12.0_f32;
        let sx = (w - box_w - 24.0) + (24.0 - (w - box_w - 24.0)) * ease;
        let sy = y_ct + (box_h + 12.0) * ease;
        let offsets: [(f32, f32); 3] = [(pad, 20.0), (pad, 38.0), (pad, 62.0)];
        for (lbl, (ox, oy)) in self.labels_snfh.iter_mut().zip(offsets.iter()) {
            lbl.x = sx + ox;
            lbl.y = sy + *oy;
        }
    }

    /// Returns true if the cursor at `my` (y in logical pixels) is in the menu bar or dropdown.
    pub(crate) fn is_menu_zone(&self, my: f64) -> bool {
        if my < MENU_BAR_H as f64 {
            return true;
        }
        if self.menu_open >= 0 {
            let drop_h = self.dropdown_height(self.menu_open);
            if my < (MENU_BAR_H + drop_h) as f64 {
                return true;
            }
        }
        false
    }
}
