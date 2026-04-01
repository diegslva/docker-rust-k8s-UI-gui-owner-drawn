use std::sync::mpsc;
use tracing::info;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::CursorIcon;

use neuroscan_core::{
    MENU_BAR_H, MENU_DROP_W, MENU_ITEM_H, MENU_SEP_H, MENU_TOP_WS, MENU_TOP_XS, TOP_CASES,
};

use crate::app::App;
use crate::app::state::{MOUSE_SENSITIVITY, PITCH_LIMIT, ZOOM_MAX, ZOOM_MIN, ZOOM_SENSITIVITY};

impl App {
    pub(super) fn window_event_inner(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("close");
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if self.menu_open >= 0 {
                        self.menu_open = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu = Vec::new();
                    }
                    self.last_interaction = std::time::Instant::now();
                    match &event.logical_key {
                        Key::Character(ch) => match ch.as_str() {
                            "i" | "I" => {
                                self.show_panel = !self.show_panel;
                            }
                            "o" | "O" => {
                                if self.dialog_rx.is_none() && !self.infer_active {
                                    let (tx, rx) = mpsc::channel();
                                    std::thread::spawn(move || {
                                        let path = rfd::FileDialog::new()
                                            .set_title("Selecionar volume NIfTI (.nii.gz)")
                                            .add_filter("NIfTI", &["gz", "nii"])
                                            .pick_file();
                                        if let Some(p) = path {
                                            let _ = tx.send(p);
                                        }
                                    });
                                    self.dialog_rx = Some(rx);
                                }
                            }
                            // MRI Slice Plane controls
                            "4" => {
                                self.slice_visible = !self.slice_visible;
                                tracing::info!(visible = self.slice_visible, "toggle slice plane");
                            }
                            "1" => {
                                self.slice_plane = crate::volume::SlicePlane::Axial;
                                tracing::info!("slice plane: axial");
                            }
                            "2" => {
                                self.slice_plane = crate::volume::SlicePlane::Coronal;
                                tracing::info!("slice plane: coronal");
                            }
                            "3" => {
                                self.slice_plane = crate::volume::SlicePlane::Sagittal;
                                tracing::info!("slice plane: sagittal");
                            }
                            "+" | "=" => {
                                self.slice_position = (self.slice_position + 0.01).min(1.0);
                            }
                            "-" => {
                                self.slice_position = (self.slice_position - 0.01).max(0.0);
                            }
                            _ => {}
                        },
                        Key::Named(NamedKey::F2) => {
                            use crate::app::state::BrainViewMode;
                            self.brain_view = match self.brain_view {
                                BrainViewMode::Transparent => BrainViewMode::TumorsOnly,
                                BrainViewMode::TumorsOnly => BrainViewMode::Transparent,
                                BrainViewMode::Opaque => BrainViewMode::TumorsOnly,
                            };
                            tracing::info!(mode = ?self.brain_view, "F2: toggle cerebro");
                        }
                        Key::Named(NamedKey::F3) => {
                            use crate::app::state::BrainViewMode;
                            self.brain_view = match self.brain_view {
                                BrainViewMode::Opaque => BrainViewMode::Transparent,
                                _ => BrainViewMode::Opaque,
                            };
                            tracing::info!(mode = ?self.brain_view, "F3: modo cerebro realista");
                        }
                        Key::Named(NamedKey::F11) => {
                            if let Some(w) = &self.window {
                                let is_fullscreen = w.fullscreen().is_some();
                                w.set_fullscreen(if is_fullscreen {
                                    None
                                } else {
                                    Some(winit::window::Fullscreen::Borderless(None))
                                });
                                tracing::info!(fullscreen = !is_fullscreen, "toggle fullscreen");
                            }
                        }
                        Key::Named(NamedKey::ArrowRight) => self.navigate_case(1),
                        Key::Named(NamedKey::ArrowLeft) => self.navigate_case(-1),
                        Key::Named(NamedKey::Escape) => {
                            // Voltar à tela inicial
                            if !self.show_home && !self.infer_active {
                                self.show_home = true;
                                let sz = PhysicalSize::new(
                                    self.gpu.as_ref().map_or(1280, |g| g.config.width),
                                    self.gpu.as_ref().map_or(720, |g| g.config.height),
                                );
                                self.build_home_labels(sz);
                            }
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Pressed {
                    let (mx, my) = self.mouse_pos.unwrap_or((0.0, 0.0));

                    // ── Click na barra de menu (top items) ────────────────────
                    let clicked_top = if my < MENU_BAR_H as f64 {
                        MENU_TOP_XS
                            .iter()
                            .zip(MENU_TOP_WS.iter())
                            .enumerate()
                            .find(|(_, (x, w))| mx >= **x as f64 && mx < (**x + **w) as f64)
                            .map_or(-1, |(i, _)| i as i32)
                    } else {
                        -1
                    };

                    if clicked_top >= 0 {
                        self.menu_open = if self.menu_open == clicked_top {
                            -1
                        } else {
                            clicked_top
                        };
                        self.menu_hover_item = -1;
                        let sz = self
                            .window
                            .as_ref()
                            .map(|w| w.inner_size())
                            .unwrap_or(PhysicalSize::new(1280, 720));
                        self.rebuild_menu_labels(sz);
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                        return;
                    }

                    // ── Click num item do dropdown ────────────────────────────
                    if self.menu_open >= 0 && self.menu_hover_item >= 0 {
                        let menu_id = self.menu_open;
                        let item_idx = self.menu_hover_item;
                        let entries = self.build_menu_entries(menu_id);
                        self.menu_open = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu = Vec::new();
                        if let Some((_, _, is_sep)) = entries.get(item_idx as usize) {
                            if !is_sep {
                                match (menu_id, item_idx) {
                                    (0, 0) => {
                                        // Arquivo → Abrir Volume NIfTI
                                        if self.dialog_rx.is_none() && !self.infer_active {
                                            let (tx, rx) = mpsc::channel();
                                            std::thread::spawn(move || {
                                                let path = rfd::FileDialog::new()
                                                    .set_title("Selecionar volume NIfTI (.nii.gz)")
                                                    .add_filter("NIfTI", &["gz", "nii"])
                                                    .pick_file();
                                                if let Some(p) = path {
                                                    let _ = tx.send(p);
                                                }
                                            });
                                            self.dialog_rx = Some(rx);
                                        }
                                    }
                                    (0, 2) => {
                                        // Arquivo → Sair
                                        event_loop.exit();
                                        return;
                                    }
                                    (1, 0) => self.navigate_case(-1),
                                    (1, 1) => self.navigate_case(1),
                                    (1, n) if n >= 3 => {
                                        let case_idx = (n - 3) as usize;
                                        if case_idx < TOP_CASES.len()
                                            && case_idx != self.current_case
                                        {
                                            use crate::app::state::{
                                                TRANSITION_DURATION, load_tumor_meshes_bg,
                                            };
                                            let _ = TRANSITION_DURATION; // used in redraw
                                            self.transition_target = case_idx;
                                            self.transition_phase = f32::EPSILON;
                                            self.spinner_angle = 0.0;
                                            self.last_interaction = std::time::Instant::now();
                                            let device = self.gpu.as_ref().unwrap().device.clone();
                                            self.loading_rx = Some(load_tumor_meshes_bg(
                                                device,
                                                TOP_CASES[case_idx],
                                            ));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                        return;
                    }

                    // ── Click fora do menu: fecha dropdown ────────────────────
                    if self.menu_open >= 0 {
                        self.menu_open = -1;
                        self.menu_hover_item = -1;
                        self.labels_menu = Vec::new();
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }

                if button == MouseButton::Left {
                    let (_, my) = self.mouse_pos.unwrap_or((0.0, 0.0));
                    let in_menu = self.is_menu_zone(my);
                    if state == ElementState::Pressed && in_menu {
                        // Click na menu bar / dropdown — não inicia drag de câmera
                    } else {
                        self.mouse_pressed = state == ElementState::Pressed;
                        self.last_interaction = std::time::Instant::now();
                        if let Some(w) = &self.window {
                            w.set_cursor(if self.mouse_pressed {
                                CursorIcon::Grabbing
                            } else {
                                CursorIcon::Grab
                            });
                        }
                        if state == ElementState::Released {
                            self.mouse_pos = None;
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let mx = position.x;
                let my = position.y;

                if self.mouse_pressed {
                    self.last_interaction = std::time::Instant::now();
                    if let Some((px, py)) = self.mouse_pos {
                        self.camera.yaw += (mx - px) as f32 * MOUSE_SENSITIVITY;
                        self.camera.pitch = (self.camera.pitch
                            - (my - py) as f32 * MOUSE_SENSITIVITY)
                            .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    }
                }
                self.mouse_pos = Some((mx, my));

                // ── Hover tracking da menu bar ─────────────────────────────
                let new_top = if my < MENU_BAR_H as f64 {
                    MENU_TOP_XS
                        .iter()
                        .zip(MENU_TOP_WS.iter())
                        .enumerate()
                        .find(|(_, (x, w))| mx >= **x as f64 && mx < (**x + **w) as f64)
                        .map_or(-1, |(i, _)| i as i32)
                } else {
                    -1
                };

                let new_item = if self.menu_open >= 0 {
                    let mid = self.menu_open as usize;
                    let drop_x = MENU_TOP_XS[mid] as f64;
                    let entries = self.build_menu_entries(self.menu_open);
                    let drop_h: f64 = entries
                        .iter()
                        .map(|(_, _, s)| if *s { MENU_SEP_H } else { MENU_ITEM_H })
                        .sum::<f32>() as f64
                        + 4.0;
                    if mx >= drop_x
                        && mx < drop_x + MENU_DROP_W as f64
                        && my >= MENU_BAR_H as f64
                        && my < MENU_BAR_H as f64 + drop_h
                    {
                        let rel = my - MENU_BAR_H as f64;
                        let mut y = 0.0_f64;
                        entries
                            .iter()
                            .enumerate()
                            .find_map(|(i, (_, _, is_sep))| {
                                let ih = if *is_sep { MENU_SEP_H } else { MENU_ITEM_H } as f64;
                                if rel >= y && rel < y + ih {
                                    Some(i as i32)
                                } else {
                                    y += ih;
                                    None
                                }
                            })
                            .unwrap_or(-1)
                    } else {
                        -1
                    }
                } else {
                    -1
                };

                if new_top != self.menu_hover_top || new_item != self.menu_hover_item {
                    self.menu_hover_top = new_top;
                    self.menu_hover_item = new_item;
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }

                let cursor = if self.is_menu_zone(my) {
                    CursorIcon::Default
                } else if self.mouse_pressed {
                    CursorIcon::Grabbing
                } else {
                    CursorIcon::Grab
                };
                if let Some(w) = &self.window {
                    w.set_cursor(cursor);
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.shift_held = modifiers.state().shift_key();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.last_interaction = std::time::Instant::now();
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                if self.shift_held && self.slice_visible {
                    // Shift + scroll: move slice plane
                    self.slice_position = (self.slice_position + scroll * 0.02).clamp(0.0, 1.0);
                } else {
                    // Scroll normal: zoom camera
                    self.camera.distance = (self.camera.distance - scroll * ZOOM_SENSITIVITY)
                        .clamp(ZOOM_MIN, ZOOM_MAX);
                }
            }

            WindowEvent::Resized(new_size) => {
                tracing::debug!(w = new_size.width, h = new_size.height, "resize");
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(new_size);
                }
                if self.splash_done {
                    self.build_labels(new_size);
                    self.update_snfh_label_positions(new_size);
                    self.rebuild_menu_labels(new_size);
                } else {
                    self.build_splash_labels(new_size);
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                self.handle_redraw(event_loop);
            }

            _ => {}
        }
    }
}
