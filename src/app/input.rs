use anyhow::{Context, Result};
use neuroscan_core::{
    CASES_DIR, MENU_BAR_H, MENU_DROP_W, MENU_ITEM_H, MENU_SEP_H, MENU_TOP_WS, MENU_TOP_XS,
    TOP_CASES,
};
use std::sync::{Arc, mpsc};
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Icon, WindowAttributes, WindowId};

use crate::renderer::{GpuState, MeshEntry, Prim2DBatch};
use crate::ui::Label;

use wgpu::Device;

use super::state::{
    App, ET_COLOR, ICON_BYTES, MOUSE_SENSITIVITY, NETC_COLOR, PITCH_LIMIT, SNFH_COLOR,
    SPLASH_FADEOUT_DURATION, TRANSITION_DURATION, TUMOR_COUNT, WINDOW_HEIGHT, WINDOW_WIDTH,
    ZOOM_MAX, ZOOM_MIN, ZOOM_SENSITIVITY, load_brain_meshes_bg, load_tumor_meshes_bg,
};

use super::state::AUTO_ROTATE_IDLE_S;
use super::state::AUTO_ROTATE_SPEED;

fn load_embedded_icon() -> Result<Icon> {
    let img = image::load_from_memory(ICON_BYTES)
        .context("falha ao decodificar icone")?
        .into_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).context("falha ao criar Icon")
}

fn center_position(event_loop: &ActiveEventLoop) -> Option<PhysicalPosition<i32>> {
    let monitor = event_loop
        .primary_monitor()
        .or_else(|| event_loop.available_monitors().next())?;
    let ms = monitor.size();
    let mp = monitor.position();
    let scale = monitor.scale_factor();
    let pw = (WINDOW_WIDTH * scale) as i32;
    let ph = (WINDOW_HEIGHT * scale) as i32;
    Some(PhysicalPosition::new(
        mp.x + (ms.width as i32 - pw) / 2,
        mp.y + (ms.height as i32 - ph) / 2,
    ))
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let icon = load_embedded_icon().ok();
        let mut attrs = WindowAttributes::default()
            .with_title("NeuroScan — Visualizador Medico 3D")
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_visible(false);
        if let Some(ic) = icon {
            attrs = attrs.with_window_icon(Some(ic));
        }

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                warn!(error = %e, "falha ao criar janela");
                event_loop.exit();
                return;
            }
        };
        if let Some(pos) = center_position(event_loop) {
            window.set_outer_position(pos);
        }

        let gpu = match pollster::block_on(GpuState::new(Arc::clone(&window))) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "falha ao inicializar wgpu");
                event_loop.exit();
                return;
            }
        };
        self.gpu = Some(gpu);

        let size = window.inner_size();
        self.build_splash_labels(size);

        // Carregar todos os meshes em thread de fundo
        // wgpu::Device is internally Arc<..> — clone is cheap
        let device: Device = self.gpu.as_ref().unwrap().device.clone();
        self.splash_rx = Some(load_brain_meshes_bg(device, TOP_CASES[0]));
        info!("splash iniciada — carregamento de meshes em background");

        self.camera.target = glam::Vec3::ZERO;
        self.camera.distance = 4.0;
        self.last_frame = std::time::Instant::now();
        self.last_interaction = std::time::Instant::now();

        // Renderizar um frame escuro ANTES de mostrar a janela — elimina flash branco do OS.
        self.window = Some(Arc::clone(&window));
        {
            let cam = self.camera.build_uniform(size.width, size.height);
            let sw = size.width as f32;
            let sh = size.height as f32;
            let mut first_prims = Prim2DBatch::new();
            first_prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, 1.0], sw, sh);
            let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
            if let Some(gpu) = &mut self.gpu {
                let empty_overlay = Prim2DBatch::new();
                if let Err(e) =
                    gpu.render(&cam, &[], &label_refs, &first_prims, &empty_overlay, &[])
                {
                    warn!(error = %e, "erro no frame inicial da splash");
                }
            }
        }
        window.set_visible(true);
        self.window_shown = true;
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                            _ => {}
                        },
                        Key::Named(NamedKey::ArrowRight) => self.navigate_case(1),
                        Key::Named(NamedKey::ArrowLeft) => self.navigate_case(-1),
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
                                            self.transition_target = case_idx;
                                            self.transition_phase = f32::EPSILON;
                                            self.spinner_angle = 0.0;
                                            self.last_interaction = std::time::Instant::now();
                                            let device: Device =
                                                self.gpu.as_ref().unwrap().device.clone();
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

            WindowEvent::MouseWheel { delta, .. } => {
                self.last_interaction = std::time::Instant::now();
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera.distance =
                    (self.camera.distance - scroll * ZOOM_SENSITIVITY).clamp(ZOOM_MIN, ZOOM_MAX);
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

    fn exiting(&mut self, _: &ActiveEventLoop) {
        info!("shutdown");
    }
}

impl App {
    pub(crate) fn handle_redraw(&mut self, event_loop: &ActiveEventLoop) {
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.05);
        self.last_frame = std::time::Instant::now();

        // ─────────────────────────────────────────────────────────────
        // SPLASH SCREEN — exibida enquanto os meshes carregam em fundo
        // ─────────────────────────────────────────────────────────────
        if !self.splash_done {
            self.splash_t += dt;
            self.spinner_angle += dt * std::f32::consts::TAU * 0.55;

            if let Some(rx) = &self.splash_rx {
                if let Ok(new_meshes) = rx.try_recv() {
                    self.meshes = new_meshes;
                    for i in 0..TUMOR_COUNT {
                        self.centroids[i] = self
                            .meshes
                            .get(i)
                            .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                    }
                    self.scan = neuroscan_core::ScanMeta::load(
                        &neuroscan_core::ScanMeta::case_path(TOP_CASES[0]),
                    );
                    self.splash_rx = None;
                    let sz = PhysicalSize::new(
                        self.gpu.as_ref().map_or(1280, |g| g.config.width),
                        self.gpu.as_ref().map_or(720, |g| g.config.height),
                    );
                    self.build_labels(sz);
                    self.update_snfh_label_positions(sz);
                    self.rebuild_menu_labels(sz);
                    info!(meshes = self.meshes.len(), "splash: meshes prontas");
                }
            }

            if self.splash_rx.is_none() {
                self.splash_fade += dt / SPLASH_FADEOUT_DURATION;
                if self.splash_fade >= 1.0 {
                    self.splash_done = true;
                    self.splash_fade = 1.0;
                    self.spinner_angle = 0.0;
                }
            }

            let sz = PhysicalSize::new(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            let sw = sz.width as f32;
            let sh = sz.height as f32;
            let cam = self.camera.build_uniform(sz.width, sz.height);

            if self.splash_rx.is_some() {
                let prims = self.build_splash_primitives(sw, sh);
                let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
                if let Some(gpu) = &mut self.gpu {
                    let empty_overlay = Prim2DBatch::new();
                    if let Err(e) = gpu.render(&cam, &[], &label_refs, &prims, &empty_overlay, &[])
                    {
                        warn!(error = %e, "erro render splash");
                    }
                }
                if !self.window_shown {
                    if let Some(w) = &self.window {
                        w.set_visible(true);
                    }
                    self.window_shown = true;
                }
            } else {
                let overlay = (1.0 - self.splash_fade).max(0.0);
                let pulse_t = self.pulse_t;
                let spinner = self.spinner_angle;
                let mut prims =
                    self.build_primitives(&cam.mvp, sw, sh, pulse_t, 0.0, spinner, false);
                if overlay > 0.01 {
                    prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, overlay], sw, sh);
                }
                let label_refs: Vec<&Label> = self.labels_always.iter().collect();
                if let Some(gpu) = &mut self.gpu {
                    let entries: Vec<MeshEntry> = self
                        .meshes
                        .iter()
                        .map(|m| MeshEntry {
                            mesh: &m.mesh,
                            tint: m.tint,
                            alpha: m.alpha,
                        })
                        .collect();
                    let empty_overlay = Prim2DBatch::new();
                    if let Err(e) =
                        gpu.render(&cam, &entries, &label_refs, &prims, &empty_overlay, &[])
                    {
                        warn!(error = %e, "erro render fade-out splash");
                    }
                }
            }

            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }

        // ── File dialog: checar se usuário selecionou arquivo ──────
        let picked_path: Option<std::path::PathBuf> = if let Some(rx) = &self.dialog_rx {
            match rx.try_recv() {
                Ok(p) => {
                    self.dialog_rx = None;
                    Some(p)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        if let Some(path) = picked_path {
            let path_str = path.display().to_string();
            let out_dir = "assets/models/infer".to_string();
            let device_bg: Device = self.gpu.as_ref().unwrap().device.clone();
            let (tx, rx) = mpsc::channel::<bool>();
            let out_clone = out_dir.clone();
            std::thread::spawn(move || {
                info!(input = %path_str, "iniciando inferencia Python");
                let status = std::process::Command::new("python")
                    .args([
                        "scripts/ml/infer_single.py",
                        "--input",
                        &path_str,
                        "--output-dir",
                        &out_clone,
                        "--model",
                        "assets/models/onnx/nnunet_brats_4ch.onnx",
                        "--meta",
                        "assets/models/brain_meta.json",
                    ])
                    .status();
                let ok = matches!(status, Ok(s) if s.success());
                if ok {
                    let defs = [
                        (format!("{}/tumor_et.obj", out_clone), ET_COLOR, 1.0_f32),
                        (format!("{}/tumor_snfh.obj", out_clone), SNFH_COLOR, 1.0_f32),
                        (format!("{}/tumor_netc.obj", out_clone), NETC_COLOR, 1.0_f32),
                    ];
                    // Pre-load on this thread; meshes not sent over channel (wgpu Buffer
                    // is not Send across all platforms). Signal ok=true; main thread reloads.
                    let _meshes: Vec<super::state::LoadedMesh> = defs
                        .iter()
                        .filter_map(|(p, t, a)| {
                            crate::mesh::Mesh::from_obj(&device_bg, p).ok().map(|m| {
                                super::state::LoadedMesh {
                                    mesh: m,
                                    tint: *t,
                                    alpha: *a,
                                }
                            })
                        })
                        .collect();
                }
                let _ = tx.send(ok);
            });
            self.infer_rx = Some(rx);
            self.infer_active = true;
            info!("inferencia iniciada");
        }

        // ── Inferência: checar conclusão ────────────────────────────
        if let Some(rx) = &self.infer_rx {
            if let Ok(ok) = rx.try_recv() {
                self.infer_rx = None;
                self.infer_active = false;
                if ok {
                    let device: Device = self.gpu.as_ref().unwrap().device.clone();
                    let out_dir = "assets/models/infer";
                    let defs = [
                        (format!("{}/tumor_et.obj", out_dir), ET_COLOR, 1.0_f32),
                        (format!("{}/tumor_snfh.obj", out_dir), SNFH_COLOR, 1.0_f32),
                        (format!("{}/tumor_netc.obj", out_dir), NETC_COLOR, 1.0_f32),
                    ];
                    let new_meshes: Vec<super::state::LoadedMesh> = defs
                        .iter()
                        .filter_map(|(p, t, a)| {
                            crate::mesh::Mesh::from_obj(&device, p).ok().map(|m| {
                                super::state::LoadedMesh {
                                    mesh: m,
                                    tint: *t,
                                    alpha: *a,
                                }
                            })
                        })
                        .collect();
                    for (i, lm) in new_meshes.into_iter().enumerate() {
                        if i < self.meshes.len() {
                            self.meshes[i] = lm;
                        }
                    }
                    for i in 0..TUMOR_COUNT {
                        self.centroids[i] = self
                            .meshes
                            .get(i)
                            .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                    }
                    let infer_meta = format!("{}/scan_meta.json", out_dir);
                    self.scan = neuroscan_core::ScanMeta::load(&infer_meta);
                    let sz = PhysicalSize::new(
                        self.gpu.as_ref().map_or(1280, |g| g.config.width),
                        self.gpu.as_ref().map_or(720, |g| g.config.height),
                    );
                    self.build_labels(sz);
                    self.update_snfh_label_positions(sz);
                    self.rebuild_menu_labels(sz);
                    info!("caso inferido carregado com sucesso");
                } else {
                    warn!("inferencia Python falhou ou foi cancelada");
                }
            }
        }

        // ── Spinner de inferência ──
        if self.infer_active {
            self.spinner_angle += dt * std::f32::consts::TAU * 0.75;
        }

        // ── Rotação automática após inatividade ──
        if self.last_interaction.elapsed().as_secs_f32() > AUTO_ROTATE_IDLE_S {
            self.camera.yaw += AUTO_ROTATE_SPEED * dt * 60.0;
        }

        // ── Pulso dos pontos de ancoragem ──
        self.pulse_t += dt;

        // ── Spinner: gira apenas enquanto há transição ativa ──
        if self.transition_phase > 0.0 {
            self.spinner_angle += dt * std::f32::consts::TAU * 0.75;
        }

        // ── Checar se a thread de carregamento terminou ──
        let mut rebuild_needed = false;
        let meshes_arrived = if let Some(rx) = &self.loading_rx {
            match rx.try_recv() {
                Ok(new_meshes) => {
                    for (i, lm) in new_meshes.into_iter().enumerate() {
                        if i < self.meshes.len() {
                            self.meshes[i] = lm;
                        }
                    }
                    for i in 0..TUMOR_COUNT {
                        self.centroids[i] = self
                            .meshes
                            .get(i)
                            .map_or(glam::Vec3::ZERO, |m| m.mesh.centroid);
                    }
                    self.current_case = self.transition_target;
                    self.scan = neuroscan_core::ScanMeta::load(
                        &neuroscan_core::ScanMeta::case_path(TOP_CASES[self.current_case]),
                    );
                    rebuild_needed = true;
                    info!(case = TOP_CASES[self.current_case], "caso carregado (bg)");
                    true
                }
                Err(_) => false,
            }
        } else {
            false
        };
        if meshes_arrived {
            self.loading_rx = None;
        }

        // ── Avanço da fase de transição ──
        if self.transition_phase > 0.0 {
            let loading_done = self.loading_rx.is_none();
            if self.transition_phase < 0.5 || loading_done {
                self.transition_phase += dt / TRANSITION_DURATION;
            }
            self.transition_phase =
                self.transition_phase
                    .min(if loading_done { 2.0 } else { 0.499 });
            if self.transition_phase >= 1.0 {
                self.transition_phase = 0.0;
                self.spinner_angle = 0.0;
            }
        }

        let size = PhysicalSize::new(
            self.gpu.as_ref().map_or(1280, |g| g.config.width),
            self.gpu.as_ref().map_or(720, |g| g.config.height),
        );

        // ── Animação do SNFH callout ──
        let snfh_target = if self.show_panel { 1.0_f32 } else { 0.0_f32 };
        let snfh_prev = self.snfh_anim_t;
        let snfh_speed = dt / 0.28;
        self.snfh_anim_t = if self.snfh_anim_t < snfh_target {
            (self.snfh_anim_t + snfh_speed).min(snfh_target)
        } else {
            (self.snfh_anim_t - snfh_speed).max(snfh_target)
        };
        if (self.snfh_anim_t - snfh_prev).abs() > 0.001 {
            self.update_snfh_label_positions(size);
        }

        if rebuild_needed {
            self.build_labels(size);
            self.update_snfh_label_positions(size);
            self.rebuild_menu_labels(size);
        }

        let cam = self.camera.build_uniform(size.width, size.height);
        let infer = self.infer_active;
        let sw = size.width as f32;
        let sh = size.height as f32;
        let prims = self.build_primitives(
            &cam.mvp,
            sw,
            sh,
            self.pulse_t,
            self.transition_phase,
            self.spinner_angle,
            infer,
        );
        let overlay_prims = self.build_menu_overlay(sw, sh);

        let mut label_refs: Vec<&Label> = self.labels_always.iter().collect();
        label_refs.extend(self.labels_snfh.iter());
        if self.show_panel {
            label_refs.extend(self.labels_panel.iter());
        }

        let mut overlay_label_refs: Vec<&Label> = self.labels_menu_bar.iter().collect();
        if self.menu_open >= 0 {
            overlay_label_refs.extend(self.labels_menu.iter());
        }

        if let Some(gpu) = &mut self.gpu {
            let entries: Vec<MeshEntry> = self
                .meshes
                .iter()
                .map(|m| MeshEntry {
                    mesh: &m.mesh,
                    tint: m.tint,
                    alpha: m.alpha,
                })
                .collect();
            if let Err(e) = gpu.render(
                &cam,
                &entries,
                &label_refs,
                &prims,
                &overlay_prims,
                &overlay_label_refs,
            ) {
                warn!(error = %e, "erro no render");
            }
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }

        // suppress unused import warning
        let _ = CASES_DIR;
        let _ = event_loop;
    }
}
