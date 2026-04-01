use tracing::{info, warn};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;

use neuroscan_core::TOP_CASES;

use crate::app::App;
use crate::app::infer::{InferMsg, InferProgress};
use crate::app::state::{
    AUTO_ROTATE_IDLE_S, AUTO_ROTATE_SPEED, ET_COLOR, INFER_FADE_DURATION, NETC_COLOR, SNFH_COLOR,
    SPLASH_FADEOUT_DURATION, TRANSITION_DURATION, TUMOR_COUNT,
};
use crate::renderer::{MeshEntry, Prim2DBatch};
use crate::ui::Label;
use neuroscan_core::CASES_DIR;

impl App {
    pub(crate) fn handle_redraw(&mut self, event_loop: &ActiveEventLoop) {
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.05);
        self.last_frame = std::time::Instant::now();

        // Tooltip fade timer
        if self.tooltip_timer > 0.0 {
            self.tooltip_timer -= dt;
            if self.tooltip_timer <= 0.0 {
                self.tooltip_text = None;
                self.labels_dirty = true;
            }
        }
        // Help overlay fade animation
        let help_speed = 4.0; // 0→1 em 0.25s
        if self.show_help {
            self.help_anim_t = (self.help_anim_t + dt * help_speed).min(1.0);
        } else {
            self.help_anim_t = (self.help_anim_t - dt * help_speed).max(0.0);
        }
        // Help fade requer rebuild de labels (alpha dos textos muda)
        if self.help_anim_t > 0.01 && self.help_anim_t < 0.99 {
            self.labels_dirty = true;
        }

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
                    if let Err(e) =
                        gpu.render(&cam, &[], &label_refs, &prims, &empty_overlay, &[], false)
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
                // Splash fade-out: elementos da splash desvanecem sob overlay escuro.
                // NÃO renderiza cena 3D — transição vai direto para home screen.
                let fade_out = self.splash_fade; // 0→1: splash desaparece para negro
                let mut prims = self.build_splash_primitives(sw, sh);
                prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, fade_out], sw, sh);
                let label_refs: Vec<&Label> = self.splash_labels.iter().collect();
                if let Some(gpu) = &mut self.gpu {
                    let empty_overlay = Prim2DBatch::new();
                    if let Err(e) =
                        gpu.render(&cam, &[], &label_refs, &prims, &empty_overlay, &[], false)
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
            // Limpar erro de inferencia anterior (permite nova tentativa)
            self.python_env_error = None;

            // Validar arquivo: rejeitar resource forks macOS (._*) e arquivos invisiveis
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if filename.starts_with("._") || filename.starts_with('.') {
                self.python_env_error = Some(format!(
                    "Arquivo invalido: '{}'. Selecione um .nii.gz sem prefixo '._'.",
                    filename
                ));
                self.show_home = true;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }

            // Pipeline nativo: sem dependencia de Python — nenhum check necessario
            let path_str = path.display().to_string();
            let out_dir = "assets/models/cases/BRATS_CUSTOM";
            info!(input = %path_str, outdir = %out_dir, "iniciando inferencia NeuroScan");
            self.show_home = false;
            let rx = crate::app::infer::pipeline::launch(&path_str, out_dir);
            self.infer_rx = Some(rx);
            self.infer_active = true;
            self.infer_fade = 0.0;
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
        // Ao receber Done(true): carrega meshes e inicia fade suave.
        // NÃO desliga infer_active imediatamente — o fade cuida disso.
        if let Some(ok) = infer_done {
            self.infer_rx = None;
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
                                texture_map_index: None,
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
                // Carregar volume MRI para slice viewer (se .npy existe)
                let vol_path = format!("{}/volume_flair.npy", out_dir);
                let meta_path = "assets/models/brain_meta.json";
                match crate::volume::VolumeData::load(&vol_path, meta_path) {
                    Ok(vol) => {
                        if let Some(gpu) = &mut self.gpu {
                            gpu.upload_volume(&vol);
                        }
                        self.volume = Some(vol);
                        info!("volume MRI carregado para slice viewer");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "volume .npy nao disponivel (slice viewer desabilitado)");
                    }
                }

                // Inicia fade suave — infer_active permanece true durante o fade
                self.infer_fade = 0.01;
                info!("caso inferido carregado — iniciando fade para 3D");
            } else {
                // Falha: captura o erro do progresso (se Python enviou NEUROSCAN:ERROR)
                // e mostra na home screen para o usuario saber o que aconteceu
                let err_msg = self
                    .infer_progress
                    .as_ref()
                    .and_then(|p| match &p.phase {
                        crate::app::infer::InferPhase::Error(e) => Some(e.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        "Falha na inferencia. Verifique o arquivo NIfTI.".to_string()
                    });
                self.infer_active = false;
                self.infer_progress = None;
                self.infer_labels.clear();
                self.infer_fade = 0.0;
                self.show_home = true;
                self.python_env_error = Some(err_msg.clone());
                warn!(error = %err_msg, "inferencia falhou — erro visivel na home screen");
            }
        }

        // ── Tela de inferência / fade para 3D ──────────────────────
        if self.infer_active {
            let sz = PhysicalSize::new(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            let sw = sz.width as f32;
            let sh = sz.height as f32;
            let cam = self.camera.build_uniform(sz.width, sz.height);

            if self.infer_fade > 0.0 {
                // ── FADE: cena 3D emergindo por baixo de overlay escuro ──
                self.infer_fade += dt / INFER_FADE_DURATION;
                if self.infer_fade >= 1.0 {
                    // Fade completo — transição concluída
                    self.infer_active = false;
                    self.infer_fade = 0.0;
                    self.infer_progress = None;
                    self.infer_labels.clear();
                    info!("fade inferencia->3D concluido");
                } else {
                    // Renderiza cena 3D com overlay escuro que dissolve
                    let fade_alpha = (1.0 - self.infer_fade).max(0.0);
                    let prims = self.build_primitives(
                        &cam.mvp,
                        sw,
                        sh,
                        self.pulse_t,
                        0.0,
                        self.spinner_angle,
                        false,
                    );
                    let mut overlay_prims = self.build_menu_overlay(sw, sh);
                    // Overlay de fade: retângulo escuro que dissolve sobre a cena 3D
                    overlay_prims.rect(0.0, 0.0, sw, sh, [0.03, 0.04, 0.08, fade_alpha], sw, sh);
                    let mut label_refs: Vec<&Label> = self.labels_always.iter().collect();
                    label_refs.extend(self.labels_snfh.iter());
                    let mut overlay_label_refs: Vec<&Label> = self.labels_menu_bar.iter().collect();
                    if self.menu_open >= 0 {
                        overlay_label_refs.extend(self.labels_menu.iter());
                    }
                    if let Some(gpu) = &mut self.gpu {
                        let entries: Vec<MeshEntry> = self
                            .meshes
                            .iter()
                            .enumerate()
                            .filter_map(|(i, m)| {
                                self.brain_view.effective_alpha(i, m.alpha).map(|a| {
                                    let is_brain = i >= crate::app::state::TUMOR_COUNT;
                                    {
                                        // Textura ativa no modo opaco (F3) para brain meshes
                                        let use_tex = if is_brain
                                            && self.brain_view
                                                == crate::app::state::BrainViewMode::Opaque
                                            && m.texture_map_index.is_some()
                                        {
                                            1.0
                                        } else {
                                            0.0
                                        };
                                        MeshEntry {
                                            mesh: &m.mesh,
                                            tint: m.tint,
                                            alpha: a,
                                            roughness: if is_brain { 0.7 } else { 0.3 },
                                            sss_strength: if is_brain { 0.15 } else { 0.0 },
                                            use_texture: use_tex,
                                            texture_index: m.texture_map_index,
                                        }
                                    }
                                })
                            })
                            .collect();
                        let sa = self.slice_visible && self.volume.is_some();
                        if let Err(e) = gpu.render(
                            &cam,
                            &entries,
                            &label_refs,
                            &prims,
                            &overlay_prims,
                            &overlay_label_refs,
                            sa,
                        ) {
                            warn!(error = %e, "erro render fade inferencia->3D");
                        }
                    }
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                    return;
                }
            } else {
                // ── INFERÊNCIA EM ANDAMENTO: tela dedicada ──
                if let Some(p) = &mut self.infer_progress {
                    p.anim_t += dt;
                    p.elapsed_secs += dt;
                }
                self.spinner_angle += dt * std::f32::consts::TAU * 0.75;
                self.build_infer_labels(sz);
                let prims = self.build_infer_primitives(sw, sh);
                let label_refs: Vec<&Label> = self.infer_labels.iter().collect();
                if let Some(gpu) = &mut self.gpu {
                    let empty_overlay = Prim2DBatch::new();
                    if let Err(e) =
                        gpu.render(&cam, &[], &label_refs, &prims, &empty_overlay, &[], false)
                    {
                        warn!(error = %e, "erro render tela inferencia");
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }
        }

        // ── HOME SCREEN: tela inicial antes de qualquer inferência ──
        if self.show_home {
            self.home_anim_t += dt;
            let sz = PhysicalSize::new(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            let sw = sz.width as f32;
            let sh = sz.height as f32;
            self.build_home_labels(sz);
            let prims = self.build_home_primitives(sw, sh);
            let label_refs: Vec<&Label> = self.home_labels.iter().collect();
            let overlay_prims = self.build_menu_overlay(sw, sh);
            // Menu bar + dropdown items (quando aberto) — mesmo padrão do render 3D
            let mut overlay_label_refs: Vec<&Label> = self.labels_menu_bar.iter().collect();
            if self.menu_open >= 0 {
                overlay_label_refs.extend(self.labels_menu.iter());
            }
            let cam = self.camera.build_uniform(sz.width, sz.height);
            if let Some(gpu) = &mut self.gpu {
                if let Err(e) = gpu.render(
                    &cam,
                    &[],
                    &label_refs,
                    &prims,
                    &overlay_prims,
                    &overlay_label_refs,
                    false,
                ) {
                    warn!(error = %e, "erro render home screen");
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

        if rebuild_needed || self.labels_dirty {
            self.build_labels(size);
            self.update_snfh_label_positions(size);
            if rebuild_needed {
                self.rebuild_menu_labels(size);
            }
            self.labels_dirty = false;
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
                .enumerate()
                .filter_map(|(i, m)| {
                    self.brain_view.effective_alpha(i, m.alpha).map(|a| {
                        let is_brain = i >= crate::app::state::TUMOR_COUNT;
                        {
                            let use_tex = if is_brain
                                && self.brain_view == crate::app::state::BrainViewMode::Opaque
                                && m.texture_map_index.is_some()
                            {
                                1.0
                            } else {
                                0.0
                            };
                            MeshEntry {
                                mesh: &m.mesh,
                                tint: m.tint,
                                alpha: a,
                                roughness: if is_brain { 0.7 } else { 0.3 },
                                sss_strength: if is_brain { 0.15 } else { 0.0 },
                                use_texture: use_tex,
                                texture_index: m.texture_map_index,
                            }
                        }
                    })
                })
                .collect();
            // MRI Slice Plane: gerar quad e atualizar buffers se ativo
            let slice_active = self.slice_visible && self.volume.is_some();
            if slice_active {
                if let Some(vol) = &self.volume {
                    let (wmin, wmax) = vol.world_bounds();
                    let (verts, idxs) = crate::mesh::generate_slice_quad(
                        self.slice_plane,
                        self.slice_position,
                        wmin,
                        wmax,
                    );
                    gpu.queue
                        .write_buffer(&gpu.slice_quad_vb, 0, bytemuck::cast_slice(&verts));
                    gpu.queue
                        .write_buffer(&gpu.slice_quad_ib, 0, bytemuck::cast_slice(&idxs));
                    let params = crate::renderer::SliceParams {
                        world_min: wmin.to_array(),
                        _pad0: 0.0,
                        world_max: wmax.to_array(),
                        alpha: 0.85,
                    };
                    gpu.queue.write_buffer(
                        &gpu.slice_params_buffer,
                        0,
                        bytemuck::cast_slice(&[params]),
                    );
                }
            }
            if let Err(e) = gpu.render(
                &cam,
                &entries,
                &label_refs,
                &prims,
                &overlay_prims,
                &overlay_label_refs,
                slice_active,
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
