use neuroscan_core::{MENU_BAR_H, MENU_DROP_W, MENU_ITEM_H, MENU_SEP_H, MENU_TOP_WS, MENU_TOP_XS};

use crate::app::App;
use crate::app::infer::InferPhase;
use crate::renderer::Prim2DBatch;

use super::{ACCENT, ACCENT_GLOW, BG_BORDER, BG_DEEP, BG_SURFACE};

use super::super::projection::project_to_screen;
use super::super::state::{ET_COLOR, NETC_COLOR, PULSE_FREQ, SNFH_COLOR};

impl App {
    /// Geometria animada da splash: fundo escuro, scan-line MRI, anéis pulsantes, arco orbital.
    pub(crate) fn build_splash_primitives(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();
        let cx = w / 2.0;
        let cy = h / 2.0 + 28.0;
        let t = self.splash_t;
        let orbit = self.spinner_angle;

        // Fundo
        b.rect(0.0, 0.0, w, h, BG_DEEP, w, h);

        // Linha de varredura horizontal — estilo scanner MRI
        let scan_y = ((t * 0.36).fract() * h).clamp(0.0, h - 2.0);
        b.rect(0.0, scan_y - 1.0, w, 2.5, [0.30, 0.58, 0.88, 0.13], w, h);
        b.rect(
            w * 0.25,
            scan_y - 0.5,
            w * 0.50,
            1.0,
            [0.52, 0.78, 1.0, 0.20],
            w,
            h,
        );

        // Três anéis pulsantes (defasados em 1/3 do ciclo)
        let ring_n = 24_usize;
        let ring_r_max = 88.0_f32;
        let cycle = 2.8_f32;
        for i in 0..3usize {
            let phase = ((t / cycle) + (i as f32 / 3.0)).fract();
            let r = phase * ring_r_max;
            let alpha = (1.0 - phase).powf(1.8) * 0.30;
            if alpha < 0.01 {
                continue;
            }
            for j in 0..ring_n {
                let a0 = (j as f32 / ring_n as f32) * std::f32::consts::TAU;
                let a1 = ((j + 1) as f32 / ring_n as f32) * std::f32::consts::TAU;
                b.line(
                    cx + r * a0.cos(),
                    cy + r * a0.sin(),
                    cx + r * a1.cos(),
                    cy + r * a1.sin(),
                    [0.22, 0.52, 0.86, alpha],
                    1.2,
                    w,
                    h,
                );
            }
        }

        // Trilha circular do arco orbital
        let orbit_r = 44.0_f32;
        for j in 0..36usize {
            let a0 = (j as f32 / 36.0) * std::f32::consts::TAU;
            let a1 = ((j + 1) as f32 / 36.0) * std::f32::consts::TAU;
            b.line(
                cx + orbit_r * a0.cos(),
                cy + orbit_r * a0.sin(),
                cx + orbit_r * a1.cos(),
                cy + orbit_r * a1.sin(),
                [0.12, 0.20, 0.36, 0.26],
                1.0,
                w,
                h,
            );
        }

        // Arco orbital com cauda gradiente
        let arc_n = 28_usize;
        let arc_span = std::f32::consts::TAU * (260.0 / 360.0);
        let tail_a = orbit - arc_span;
        for j in 0..arc_n {
            let t0 = j as f32 / arc_n as f32;
            let t1 = (j + 1) as f32 / arc_n as f32;
            let a0 = tail_a + t0 * arc_span;
            let a1 = tail_a + t1 * arc_span;
            let br = t0.powf(1.3);
            b.line(
                cx + orbit_r * a0.cos(),
                cy + orbit_r * a0.sin(),
                cx + orbit_r * a1.cos(),
                cy + orbit_r * a1.sin(),
                [0.42 + 0.22 * br, 0.65 + 0.16 * br, 1.0, br * 0.88],
                2.0,
                w,
                h,
            );
        }

        // Ponto brilhante na cabeca do arco
        let hx = cx + orbit_r * orbit.cos();
        let hy = cy + orbit_r * orbit.sin();
        let hr = 2.6_f32;
        b.rect(
            hx - hr,
            hy - hr,
            hr * 2.0,
            hr * 2.0,
            [0.70, 0.90, 1.0, 0.92],
            w,
            h,
        );

        // Linhas decorativas horizontais acima e abaixo do titulo
        let title_y = cy - 122.0;
        let line_col = [0.25, 0.45, 0.70, 0.22];
        let line_w = w * 0.45;
        let line_x = (w - line_w) / 2.0;
        b.rect(line_x, title_y - 12.0, line_w, 1.0, line_col, w, h);
        b.rect(line_x, title_y + 80.0, line_w, 1.0, line_col, w, h);

        b
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_primitives(
        &self,
        mvp: &[[f32; 4]; 4],
        w: f32,
        h: f32,
        pulse_t: f32,
        transition: f32,
        spinner_angle: f32,
        infer_active: bool,
    ) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();
        let y_ct = h * 0.14;
        let box_w = (w * 0.175).clamp(190.0, 240.0);
        let box_h = 92.0_f32;
        let bg = BG_SURFACE;
        let inner_ov = [0.20, 0.26, 0.40, 0.09_f32];
        let dot_r = 3.5 + 1.5 * (pulse_t * std::f32::consts::TAU * PULSE_FREQ).sin().abs();

        // ET callout (top-left)
        let et_x = 24.0;
        let et_y = y_ct;
        b.rect(et_x, et_y, box_w, box_h, bg, w, h);
        b.rect(et_x, et_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
        b.rect(
            et_x,
            et_y,
            box_w,
            2.0,
            [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 1.0],
            w,
            h,
        );
        if let Some((cx, cy)) = project_to_screen(self.centroids[0], mvp, w, h) {
            b.line(
                et_x + box_w,
                et_y + box_h * 0.5,
                cx,
                cy,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.60],
                1.5,
                w,
                h,
            );
            b.rect(
                cx - dot_r,
                cy - dot_r,
                dot_r * 2.0,
                dot_r * 2.0,
                [ET_COLOR[0], ET_COLOR[1], ET_COLOR[2], 0.90],
                w,
                h,
            );
        }

        // SNFH callout — posição interpolada suavemente
        let ease_snfh = {
            let t = self.snfh_anim_t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let snfh_x = (w - box_w - 24.0) + (24.0 - (w - box_w - 24.0)) * ease_snfh;
        let snfh_y = y_ct + (y_ct + box_h + 12.0 - y_ct) * ease_snfh;
        let snfh_line_x = snfh_x + box_w * ease_snfh;
        b.rect(snfh_x, snfh_y, box_w, box_h, bg, w, h);
        b.rect(snfh_x, snfh_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
        b.rect(
            snfh_x,
            snfh_y,
            box_w,
            2.0,
            [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 1.0],
            w,
            h,
        );
        if let Some((cx, cy)) = project_to_screen(self.centroids[1], mvp, w, h) {
            b.line(
                snfh_line_x,
                snfh_y + box_h * 0.5,
                cx,
                cy,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.60],
                1.5,
                w,
                h,
            );
            b.rect(
                cx - dot_r,
                cy - dot_r,
                dot_r * 2.0,
                dot_r * 2.0,
                [SNFH_COLOR[0], SNFH_COLOR[1], SNFH_COLOR[2], 0.90],
                w,
                h,
            );
        }

        // NETC callout (bottom-center)
        let netc_x = (w / 2.0 - box_w / 2.0).max(24.0);
        let netc_y = (h - 185.0).max(y_ct + box_h + 20.0);
        b.rect(netc_x, netc_y, box_w, box_h, bg, w, h);
        b.rect(netc_x, netc_y + 2.0, box_w, box_h - 2.0, inner_ov, w, h);
        b.rect(
            netc_x,
            netc_y,
            box_w,
            2.0,
            [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 1.0],
            w,
            h,
        );
        if let Some((cx, cy)) = project_to_screen(self.centroids[2], mvp, w, h) {
            b.line(
                netc_x + box_w * 0.5,
                netc_y,
                cx,
                cy,
                [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 0.60],
                1.5,
                w,
                h,
            );
            b.rect(
                cx - dot_r,
                cy - dot_r,
                dot_r * 2.0,
                dot_r * 2.0,
                [NETC_COLOR[0], NETC_COLOR[1], NETC_COLOR[2], 0.90],
                w,
                h,
            );
        }

        // Fundo do painel clínico
        if self.show_panel {
            let px = (w - 268.0).max(w * 0.72);
            let py = h * 0.14 - 8.0;
            let pw = 258.0_f32;
            let ph = h * 0.80;
            b.rect(px, py, pw, ph, [0.03, 0.05, 0.09, 0.90], w, h);
            b.rect(px, py, 2.0, ph, [0.35, 0.55, 0.85, 0.40], w, h);
        }

        // Overlay de inferência real — spinner grande centralizado
        if infer_active {
            b.rect(
                0.0,
                0.0,
                w,
                h,
                [BG_DEEP[0], BG_DEEP[1], BG_DEEP[2], 0.72],
                w,
                h,
            );
            let cx = w / 2.0;
            let cy = h / 2.0;
            let r = 52.0_f32;
            for i in 0..48usize {
                let a0 = (i as f32 / 48.0) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / 48.0) * std::f32::consts::TAU;
                b.line(
                    cx + r * a0.cos(),
                    cy + r * a0.sin(),
                    cx + r * a1.cos(),
                    cy + r * a1.sin(),
                    [0.16, 0.24, 0.40, 0.28],
                    1.2,
                    w,
                    h,
                );
            }
            let arc_span = std::f32::consts::TAU * (250.0 / 360.0);
            let tail_a = spinner_angle - arc_span;
            for j in 0..36usize {
                let t0 = j as f32 / 36.0;
                let t1 = (j + 1) as f32 / 36.0;
                let a0 = tail_a + t0 * arc_span;
                let a1 = tail_a + t1 * arc_span;
                let br = t0.powf(1.3);
                b.line(
                    cx + r * a0.cos(),
                    cy + r * a0.sin(),
                    cx + r * a1.cos(),
                    cy + r * a1.sin(),
                    [0.48 + 0.18 * br, 0.70 + 0.12 * br, 1.0, br * 0.90],
                    2.5,
                    w,
                    h,
                );
            }
            let hx = cx + r * spinner_angle.cos();
            let hy = cy + r * spinner_angle.sin();
            let hr = 3.2_f32;
            b.rect(
                hx - hr,
                hy - hr,
                hr * 2.0,
                hr * 2.0,
                [0.72, 0.90, 1.0, 0.96],
                w,
                h,
            );
        }

        // Overlay de transição — dissolve sinusoidal suave
        if transition > 0.0 {
            let alpha = (transition * std::f32::consts::PI).sin() * 0.92;
            b.rect(
                0.0,
                0.0,
                w,
                h,
                [BG_DEEP[0], BG_DEEP[1], BG_DEEP[2], alpha],
                w,
                h,
            );

            let spinner_alpha = ((transition * std::f32::consts::PI).sin() * 1.6).min(1.0);
            if spinner_alpha > 0.05 {
                let cx = w / 2.0;
                let cy = h / 2.0;
                let r = 36.0_f32;

                let track_n = 48_usize;
                for i in 0..track_n {
                    let a0 = (i as f32 / track_n as f32) * std::f32::consts::TAU;
                    let a1 = ((i + 1) as f32 / track_n as f32) * std::f32::consts::TAU;
                    b.line(
                        cx + r * a0.cos(),
                        cy + r * a0.sin(),
                        cx + r * a1.cos(),
                        cy + r * a1.sin(),
                        [0.18, 0.26, 0.42, 0.28 * spinner_alpha],
                        1.0,
                        w,
                        h,
                    );
                }

                let arc_n = 32_usize;
                let arc_span = std::f32::consts::TAU * (240.0 / 360.0);
                let tail_a = spinner_angle - arc_span;
                for i in 0..arc_n {
                    let t0 = i as f32 / arc_n as f32;
                    let t1 = (i + 1) as f32 / arc_n as f32;
                    let a0 = tail_a + t0 * arc_span;
                    let a1 = tail_a + t1 * arc_span;
                    let brightness = t0.powf(1.4);
                    let seg_alpha = brightness * 0.92 * spinner_alpha;
                    b.line(
                        cx + r * a0.cos(),
                        cy + r * a0.sin(),
                        cx + r * a1.cos(),
                        cy + r * a1.sin(),
                        [
                            0.50 + 0.15 * brightness,
                            0.72 + 0.10 * brightness,
                            1.0,
                            seg_alpha,
                        ],
                        2.2,
                        w,
                        h,
                    );
                }

                let hx = cx + r * spinner_angle.cos();
                let hy = cy + r * spinner_angle.sin();
                let hr = 2.8 * spinner_alpha;
                b.rect(
                    hx - hr,
                    hy - hr,
                    hr * 2.0,
                    hr * 2.0,
                    [0.75, 0.92, 1.0, 0.95 * spinner_alpha],
                    w,
                    h,
                );
            }
        }

        // --- Callout box de medicao (fundo + borda dourada) ---
        if self.measure_active && self.measure_point_a.is_some() && self.measure_point_b.is_some() {
            let cx = w - 260.0;
            let cy = h * 0.38;
            let bw = 240.0;
            let bh = 82.0;
            // Fundo
            b.rect(cx, cy, bw, bh, BG_SURFACE, w, h);
            // Overlay sutil
            b.rect(cx, cy, bw, bh, [0.25, 0.20, 0.10, 0.08], w, h);
            // Borda superior dourada
            b.rect(cx, cy, bw, 2.0, [1.0, 0.78, 0.30, 0.90], w, h);
        }

        // --- Marcadores de medicao (pontos + linha) ---
        if self.measure_active || self.measure_point_b.is_some() {
            let cam_u = self.camera.build_uniform(
                self.gpu.as_ref().map_or(1280, |g| g.config.width),
                self.gpu.as_ref().map_or(720, |g| g.config.height),
            );
            let marker_color = [1.0, 0.35, 0.20, 0.95]; // vermelho-alaranjado
            let line_color = [1.0, 1.0, 1.0, 0.80]; // branco

            // Ponto A
            if let Some(a) = &self.measure_point_a {
                if let Some((ax, ay)) =
                    crate::app::projection::project_to_screen(a.world_pos, &cam_u.mvp, w, h)
                {
                    let r = 5.0;
                    b.rect(ax - r, ay - r, r * 2.0, r * 2.0, marker_color, w, h);
                    // Cruz
                    b.rect(
                        ax - 1.0,
                        ay - r - 2.0,
                        2.0,
                        r * 2.0 + 4.0,
                        marker_color,
                        w,
                        h,
                    );
                    b.rect(
                        ax - r - 2.0,
                        ay - 1.0,
                        r * 2.0 + 4.0,
                        2.0,
                        marker_color,
                        w,
                        h,
                    );

                    // Ponto B + linha
                    if let Some(bp) = &self.measure_point_b {
                        if let Some((bx, by)) = crate::app::projection::project_to_screen(
                            bp.world_pos,
                            &cam_u.mvp,
                            w,
                            h,
                        ) {
                            // Ponto B
                            b.rect(bx - r, by - r, r * 2.0, r * 2.0, marker_color, w, h);
                            b.rect(
                                bx - 1.0,
                                by - r - 2.0,
                                2.0,
                                r * 2.0 + 4.0,
                                marker_color,
                                w,
                                h,
                            );
                            b.rect(
                                bx - r - 2.0,
                                by - 1.0,
                                r * 2.0 + 4.0,
                                2.0,
                                marker_color,
                                w,
                                h,
                            );
                            // Linha A-B
                            b.line(ax, ay, bx, by, line_color, 2.0, w, h);
                        }
                    }
                }
            }
        }

        // --- Help overlay background (fade-in/out) ---
        if self.help_anim_t > 0.01 {
            let alpha = self.help_anim_t * 0.88;
            b.rect(
                0.0,
                0.0,
                w,
                h,
                [BG_DEEP[0], BG_DEEP[1], BG_DEEP[2], alpha],
                w,
                h,
            );
        }

        b
    }

    /// Tela inicial (home screen): fundo escuro, scan-lines sutis, esboço
    /// cerebral pulsante e orbital arc — mesma linguagem visual da splash e
    /// inferência, mas sem indicadores de progresso.
    pub(crate) fn build_home_primitives(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();
        let t = self.home_anim_t;
        let cx = w / 2.0;
        let cy = h * 0.40;

        // Fundo escuro
        b.rect(0.0, 0.0, w, h, BG_DEEP, w, h);

        // Scan-lines sutis (efeito MRI ambiente)
        let scan_speeds = [0.14_f32, 0.22, 0.09];
        let scan_alphas = [0.032_f32, 0.022, 0.015];
        for (i, (&speed, &alpha)) in scan_speeds.iter().zip(scan_alphas.iter()).enumerate() {
            let offset = i as f32 * (h / 3.0);
            let y = ((t * speed + offset / h).fract() * h).clamp(0.0, h - 1.5);
            b.rect(0.0, y - 1.0, w, 2.5, [0.30, 0.58, 0.88, alpha], w, h);
        }

        // Esboço cerebral pulsante (dois anéis concêntricos)
        let base_r = 72.0_f32;
        let pulse_r = base_r + 10.0 * (t * 1.6).sin().abs();
        let ring_n = 48_usize;
        for j in 0..ring_n {
            let a0 = (j as f32 / ring_n as f32) * std::f32::consts::TAU;
            let a1 = ((j + 1) as f32 / ring_n as f32) * std::f32::consts::TAU;
            b.line(
                cx + pulse_r * a0.cos(),
                cy + pulse_r * a0.sin(),
                cx + pulse_r * a1.cos(),
                cy + pulse_r * a1.sin(),
                [0.12, 0.30, 0.55, 0.18],
                1.2,
                w,
                h,
            );
        }
        // Anel interno
        let inner_r = pulse_r * 0.55;
        let inner_n = 32_usize;
        for j in 0..inner_n {
            let a0 = (j as f32 / inner_n as f32) * std::f32::consts::TAU;
            let a1 = ((j + 1) as f32 / inner_n as f32) * std::f32::consts::TAU;
            b.line(
                cx + inner_r * a0.cos(),
                cy + inner_r * a0.sin(),
                cx + inner_r * a1.cos(),
                cy + inner_r * a1.sin(),
                [0.10, 0.25, 0.48, 0.12],
                0.8,
                w,
                h,
            );
        }

        // Orbital arc com tail gradient
        let orbit_r = pulse_r + 16.0;
        let arc_segs = 40_usize;
        let arc_span = 260.0_f32.to_radians();
        let arc_base = t * 1.8;
        for s in 0..arc_segs {
            let t0 = s as f32 / arc_segs as f32;
            let t1 = (s + 1) as f32 / arc_segs as f32;
            let a0 = arc_base + t0 * arc_span;
            let a1 = arc_base + t1 * arc_span;
            let alpha = 0.22 * t0.powf(1.5);
            b.line(
                cx + orbit_r * a0.cos(),
                cy + orbit_r * a0.sin(),
                cx + orbit_r * a1.cos(),
                cy + orbit_r * a1.sin(),
                [0.35, 0.65, 1.0, alpha],
                1.5,
                w,
                h,
            );
        }

        // Ponto brilhante na ponta do arco
        let tip_angle = arc_base + arc_span;
        let tip_x = cx + orbit_r * tip_angle.cos();
        let tip_y = cy + orbit_r * tip_angle.sin();
        b.rect(
            tip_x - 2.5,
            tip_y - 2.5,
            5.0,
            5.0,
            [0.55, 0.82, 1.0, 0.65],
            w,
            h,
        );

        // Linhas decorativas horizontais (estilo Stripe divider)
        let line_col = [0.25, 0.45, 0.70, 0.20];
        let line_w = w * 0.50;
        let line_x = (w - line_w) / 2.0;

        // Acima do titulo
        b.rect(line_x, h * 0.11, line_w, 1.0, line_col, w, h);
        // Abaixo do subtitulo
        b.rect(line_x, h * 0.12 + 76.0, line_w, 1.0, line_col, w, h);

        b
    }

    /// Tela de inferência: fundo escuro, scan-lines animadas, barra de progresso,
    /// barras de volume por classe e esboço pulsante do cérebro.
    pub(crate) fn build_infer_primitives(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();

        let Some(progress) = &self.infer_progress else {
            // Sem progresso disponível: tela mínima de espera
            b.rect(0.0, 0.0, w, h, BG_DEEP, w, h);
            return b;
        };

        let anim_t = progress.anim_t;
        let cx = w / 2.0;
        let cy = h / 2.0;

        // Linhas decorativas horizontais (consistencia visual com splash e home)
        let line_col = [0.25, 0.45, 0.70, 0.18];
        let line_w = w * 0.50;
        let line_x = (w - line_w) / 2.0;
        b.rect(line_x, h * 0.14, line_w, 1.0, line_col, w, h);
        b.rect(line_x, h * 0.15 + 68.0, line_w, 1.0, line_col, w, h);

        // ── Fundo escuro ────────────────────────────────────────────
        b.rect(0.0, 0.0, w, h, BG_DEEP, w, h);

        // ── Scan-lines animadas (três velocidades diferentes) ───────
        let scan_speeds = [0.18_f32, 0.27, 0.11];
        let scan_alphas = [0.038_f32, 0.028, 0.018];
        for (i, (&speed, &alpha)) in scan_speeds.iter().zip(scan_alphas.iter()).enumerate() {
            let offset = i as f32 * (h / 3.0);
            let y = ((anim_t * speed + offset / h).fract() * h).clamp(0.0, h - 1.5);
            b.rect(0.0, y - 1.0, w, 2.5, [0.30, 0.58, 0.88, alpha], w, h);
            b.rect(
                w * 0.20,
                y - 0.5,
                w * 0.60,
                1.0,
                [0.50, 0.75, 1.0, alpha * 1.5],
                w,
                h,
            );
        }

        // ── Esboço circular pulsante do cérebro (centro) ────────────
        let base_r = 62.0_f32;
        let pulse_r = base_r + 12.0 * (anim_t * 2.0).sin().abs();
        let ring_n = 48_usize;
        for j in 0..ring_n {
            let a0 = (j as f32 / ring_n as f32) * std::f32::consts::TAU;
            let a1 = ((j + 1) as f32 / ring_n as f32) * std::f32::consts::TAU;
            b.line(
                cx + pulse_r * a0.cos(),
                cy + pulse_r * a0.sin(),
                cx + pulse_r * a1.cos(),
                cy + pulse_r * a1.sin(),
                [0.15, 0.35, 0.65, 0.22],
                1.2,
                w,
                h,
            );
        }
        // Segundo anel interno mais brilhante
        let inner_r = pulse_r * 0.70;
        for j in 0..32_usize {
            let a0 = (j as f32 / 32.0) * std::f32::consts::TAU;
            let a1 = ((j + 1) as f32 / 32.0) * std::f32::consts::TAU;
            b.line(
                cx + inner_r * a0.cos(),
                cy + inner_r * a0.sin(),
                cx + inner_r * a1.cos(),
                cy + inner_r * a1.sin(),
                [0.18, 0.42, 0.75, 0.14],
                1.0,
                w,
                h,
            );
        }

        // ── Spinner orbital (arco com cauda) ────────────────────────
        let orbit_r = pulse_r + 18.0;
        let arc_span = std::f32::consts::TAU * (260.0 / 360.0);
        let tail_a = self.spinner_angle - arc_span;
        for j in 0..36_usize {
            let t0 = j as f32 / 36.0;
            let t1 = (j + 1) as f32 / 36.0;
            let a0 = tail_a + t0 * arc_span;
            let a1 = tail_a + t1 * arc_span;
            let br = t0.powf(1.3);
            b.line(
                cx + orbit_r * a0.cos(),
                cy + orbit_r * a0.sin(),
                cx + orbit_r * a1.cos(),
                cy + orbit_r * a1.sin(),
                [0.42 + 0.22 * br, 0.65 + 0.16 * br, 1.0, br * 0.80],
                2.0,
                w,
                h,
            );
        }
        // Ponto brilhante na cabeça do arco
        let hx = cx + orbit_r * self.spinner_angle.cos();
        let hy = cy + orbit_r * self.spinner_angle.sin();
        let hr = 2.8_f32;
        b.rect(
            hx - hr,
            hy - hr,
            hr * 2.0,
            hr * 2.0,
            [0.70, 0.90, 1.0, 0.92],
            w,
            h,
        );

        // ── Linha varrendo fatias (sweep line) ──────────────────────
        if progress.total_slices > 0 {
            let frac = progress.current_slice as f32 / progress.total_slices as f32;
            let sweep_area_top = h * 0.18;
            let sweep_area_h = h * 0.60;
            let sweep_y = (sweep_area_top + frac * sweep_area_h)
                .clamp(sweep_area_top, sweep_area_top + sweep_area_h);
            b.rect(
                w * 0.10,
                sweep_y - 0.5,
                w * 0.80,
                1.5,
                [ACCENT[0], ACCENT[1], ACCENT[2], 0.30],
                w,
                h,
            );
        }

        // ── Barra de progresso principal ────────────────────────────
        let bar_y = h * 0.84;
        let bar_h = 8.0_f32;
        let bar_x = w * 0.12;
        let bar_w = w * 0.76;

        // Fundo da barra (mais sutil)
        b.rect(
            bar_x,
            bar_y,
            bar_w,
            bar_h,
            [BG_SURFACE[0], BG_SURFACE[1], BG_SURFACE[2], 0.70],
            w,
            h,
        );
        // Borda superior fina
        b.rect(
            bar_x,
            bar_y,
            bar_w,
            1.0,
            [BG_BORDER[0], BG_BORDER[1], BG_BORDER[2], 0.30],
            w,
            h,
        );
        // Borda inferior fina
        b.rect(
            bar_x,
            bar_y + bar_h - 1.0,
            bar_w,
            1.0,
            [BG_BORDER[0], BG_BORDER[1], BG_BORDER[2], 0.20],
            w,
            h,
        );

        // Preenchimento proporcional ao progresso
        let frac = if progress.total_slices > 0 {
            match &progress.phase {
                InferPhase::PythonSetup => 0.01,
                InferPhase::Preprocessing => 0.02,
                InferPhase::Slicing => {
                    0.05 + 0.80 * (progress.current_slice as f32 / progress.total_slices as f32)
                }
                InferPhase::MarchingCubes => 0.90,
                InferPhase::Done => 1.0,
                InferPhase::Error(_) => 0.0,
            }
        } else {
            0.0
        };

        if frac > 0.0 {
            let fill_w = (bar_w * frac).max(bar_h);
            // Glow suave atras do fill
            b.rect(bar_x, bar_y - 2.0, fill_w, bar_h + 4.0, ACCENT_GLOW, w, h);
            // Fill principal
            b.rect(bar_x, bar_y, fill_w, bar_h, ACCENT, w, h);

            // Glow pulsante na ponta da barra (edge pulse)
            let pulse = (progress.anim_t * 3.5).sin().abs();
            let tip_w = 12.0_f32;
            let tip_x = bar_x + fill_w - tip_w * 0.5;
            b.rect(
                tip_x.max(bar_x),
                bar_y - 3.0,
                tip_w,
                bar_h + 6.0,
                [0.40, 0.80, 1.0, 0.15 + 0.20 * pulse],
                w,
                h,
            );
        }

        b
    }

    /// Constrói o batch de overlay do menu bar (Pass 2, acima do texto de cena).
    pub(crate) fn build_menu_overlay(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();

        b.rect(
            0.0,
            0.0,
            w,
            MENU_BAR_H,
            [BG_SURFACE[0], BG_SURFACE[1], BG_SURFACE[2], 0.97],
            w,
            h,
        );
        b.rect(
            0.0,
            MENU_BAR_H - 1.0,
            w,
            1.0,
            [0.18, 0.26, 0.44, 0.45],
            w,
            h,
        );

        for i in 0..3_usize {
            let ix = MENU_TOP_XS[i];
            let iw = MENU_TOP_WS[i];
            let is_open = self.menu_open == i as i32;
            let is_hovered = self.menu_hover_top == i as i32;
            if is_open {
                b.rect(ix, 0.0, iw, MENU_BAR_H, [0.20, 0.32, 0.55, 0.82], w, h);
            } else if is_hovered {
                b.rect(ix, 0.0, iw, MENU_BAR_H, [0.14, 0.22, 0.40, 0.55], w, h);
            }
        }

        if self.menu_open >= 0 {
            let mid = self.menu_open as usize;
            let drop_x = MENU_TOP_XS[mid];
            let drop_h = self.dropdown_height(self.menu_open);
            let entries = self.build_menu_entries(self.menu_open);

            b.rect(
                drop_x,
                MENU_BAR_H,
                MENU_DROP_W,
                drop_h,
                [0.05, 0.08, 0.15, 0.97],
                w,
                h,
            );
            b.rect(
                drop_x,
                MENU_BAR_H,
                MENU_DROP_W,
                1.0,
                [0.22, 0.32, 0.52, 0.55],
                w,
                h,
            );
            b.rect(
                drop_x,
                MENU_BAR_H + drop_h,
                MENU_DROP_W,
                1.0,
                [0.22, 0.32, 0.52, 0.55],
                w,
                h,
            );
            b.rect(
                drop_x,
                MENU_BAR_H,
                1.0,
                drop_h,
                [0.22, 0.32, 0.52, 0.55],
                w,
                h,
            );
            b.rect(
                drop_x + MENU_DROP_W - 1.0,
                MENU_BAR_H,
                1.0,
                drop_h,
                [0.22, 0.32, 0.52, 0.55],
                w,
                h,
            );

            let mut iy = MENU_BAR_H;
            for (idx, (_, _, is_sep)) in entries.iter().enumerate() {
                if *is_sep {
                    let sep_y = iy + MENU_SEP_H / 2.0;
                    b.rect(
                        drop_x + 8.0,
                        sep_y,
                        MENU_DROP_W - 16.0,
                        1.0,
                        [0.18, 0.26, 0.42, 0.38],
                        w,
                        h,
                    );
                    iy += MENU_SEP_H;
                } else {
                    if self.menu_hover_item == idx as i32 {
                        b.rect(
                            drop_x + 1.0,
                            iy,
                            MENU_DROP_W - 2.0,
                            MENU_ITEM_H,
                            [0.16, 0.26, 0.46, 0.80],
                            w,
                            h,
                        );
                    }
                    iy += MENU_ITEM_H;
                }
            }
        }

        b
    }
}
