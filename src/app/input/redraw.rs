use tracing::{info, warn};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;

use neuroscan_core::TOP_CASES;

use crate::app::App;
use crate::app::infer::{InferMsg, InferProgress};
use crate::app::state::{
    AUTO_ROTATE_IDLE_S, AUTO_ROTATE_SPEED, ET_COLOR, NETC_COLOR, SNFH_COLOR,
    SPLASH_FADEOUT_DURATION, TRANSITION_DURATION, TUMOR_COUNT,
};
use crate::renderer::{MeshEntry, Prim2DBatch};
use crate::ui::Label;
use neuroscan_core::CASES_DIR;

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
            let out_dir = "assets/models/cases/BRATS_CUSTOM";
            info!(input = %path_str, outdir = %out_dir, "iniciando inferencia NeuroScan");
            let rx = crate::app::infer::pipeline::launch(&path_str, out_dir);
            self.infer_rx = Some(rx);
            self.infer_active = true;
            self.infer_progress = Some(InferProgress::default());
            let sz = PhysicalSize::new(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            self.build_infer_labels(sz);
        }

        // ── Inferência: drena todas as mensagens pendentes do canal ────
        let mut infer_done: Option<bool> = None;
        if let Some(rx) = &self.infer_rx {
            loop {
                match rx.try_recv() {
                    Ok(InferMsg::Done(ok)) => {
                        infer_done = Some(ok);
                        break;
                    }
                    Ok(ref msg) => {
                        if let Some(p) = &mut self.infer_progress {
                            p.apply(msg);
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        // ── Conclusão da inferência ─────────────────────────────────
        if let Some(ok) = infer_done {
            self.infer_rx = None;
            self.infer_active = false;
            self.infer_progress = None;
            self.infer_labels.clear();
            if ok {
                let device = self.gpu.as_ref().unwrap().device.clone();
                let out_dir = "assets/models/cases/BRATS_CUSTOM";
                let defs = [
                    (format!("{}/tumor_et.obj", out_dir), ET_COLOR, 1.0_f32),
                    (format!("{}/tumor_snfh.obj", out_dir), SNFH_COLOR, 1.0_f32),
                    (format!("{}/tumor_netc.obj", out_dir), NETC_COLOR, 1.0_f32),
                ];
                let new_meshes: Vec<crate::app::state::LoadedMesh> = defs
                    .iter()
                    .filter_map(|(p, t, a)| {
                        crate::mesh::Mesh::from_obj(&device, p).ok().map(|m| {
                            crate::app::state::LoadedMesh {
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

        // ── Tela de inferência: anim_t tick + render dedicado ──────
        if self.infer_active {
            if let Some(p) = &mut self.infer_progress {
                p.anim_t += dt;
                p.elapsed_secs += dt;
            }
            self.spinner_angle += dt * std::f32::consts::TAU * 0.75;

            let sz = PhysicalSize::new(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            let sw = sz.width as f32;
            let sh = sz.height as f32;
            // Reconstrói labels de inferência a cada frame (counter de fatia muda rápido)
            self.build_infer_labels(sz);
            let prims = self.build_infer_primitives(sw, sh);
            let label_refs: Vec<&Label> = self.infer_labels.iter().collect();
            let cam = self.camera.build_uniform(sz.width, sz.height);
            if let Some(gpu) = &mut self.gpu {
                let empty_overlay = Prim2DBatch::new();
                if let Err(e) = gpu.render(&cam, &[], &label_refs, &prims, &empty_overlay, &[]) {
                    warn!(error = %e, "erro render tela inferencia");
                }
            }
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
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
        let sw = size.width as f32;
        let sh = size.height as f32;
        // infer_active is always false here — active inference returns early above
        let prims = self.build_primitives(
            &cam.mvp,
            sw,
            sh,
            self.pulse_t,
            self.transition_phase,
            self.spinner_angle,
            false,
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

        // suppress unused import warning — CASES_DIR used by other modules
        let _ = CASES_DIR;
        let _ = event_loop;
    }
}
