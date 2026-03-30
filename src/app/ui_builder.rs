use neuroscan_core::{MENU_BAR_H, MENU_TOP_XS, TOP_CASES};
use winit::dpi::PhysicalSize;

use crate::renderer::Prim2DBatch;
use crate::ui::{Color, Label};

use super::projection::project_to_screen;
use super::state::{
    App, ET_COLOR, NETC_COLOR, PULSE_FREQ, SNFH_COLOR, SPLASH_FADEOUT_DURATION, TUMOR_COUNT,
};

// ---------------------------------------------------------------------------
// Color palette helpers
// ---------------------------------------------------------------------------

pub(crate) fn col_header() -> Color {
    Color::rgb(226, 232, 240)
}

pub(crate) fn col_dim() -> Color {
    Color::rgb(94, 111, 133)
}

pub(crate) fn col_value() -> Color {
    Color::rgb(203, 213, 225)
}

pub(crate) fn col_section() -> Color {
    Color::rgb(71, 85, 105)
}

#[allow(dead_code)]
pub(crate) fn col_sep() -> [f32; 4] {
    [0.12, 0.17, 0.26, 0.70]
}

pub(crate) fn rgb_f(c: [f32; 3]) -> Color {
    Color::rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// Splash
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_splash_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let cy = h / 2.0 + 28.0;
        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(fs, "NeuroScan", 50.0, Color::rgb(222, 234, 248), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = cy - 122.0;

        let mut sub = Label::new(
            fs,
            "Visualizador Médico 3D",
            15.5,
            Color::rgb(88, 128, 168),
            0.0,
            0.0,
        );
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 5.0;

        self.splash_labels = vec![title, sub];
    }

    /// Geometria animada da splash: fundo escuro, scan-line MRI, anéis pulsantes, arco orbital.
    pub(crate) fn build_splash_primitives(&self, w: f32, h: f32) -> Prim2DBatch {
        let mut b = Prim2DBatch::new();
        let cx = w / 2.0;
        let cy = h / 2.0 + 28.0;
        let t = self.splash_t;
        let orbit = self.spinner_angle;

        // Fundo
        b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, 1.0], w, h);

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

        // Ponto brilhante na cabeça do arco
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

        b
    }
}

// ---------------------------------------------------------------------------
// Main scene labels + primitives
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_labels(&mut self, size: PhysicalSize<u32>) {
        // Snapshot all data fields needed before borrowing self.gpu mutably.
        let w = size.width as f32;
        let h = size.height as f32;
        let scan = self.scan.clone();
        let show_panel = self.show_panel;
        let current_case = self.current_case;

        let Some(gpu) = &mut self.gpu else { return };
        let fs = gpu.font_system_mut();

        let mut always: Vec<Label> = Vec::new();
        let mut panel: Vec<Label> = Vec::new();
        let mut snfh: Vec<Label> = Vec::new();
        // Labels do topo da barra ficam no overlay (Pass 2), NÃO em `always` (Pass 1).
        let mut menu_bar: Vec<Label> = Vec::new();

        // ── Barra de menu — itens do topo ────────────────────────────────────
        {
            let menu_col = Color::rgb(236, 242, 255);
            let bar_y = (MENU_BAR_H - 11.5_f32 * 1.25) / 2.0;
            let tops = [
                ("Arquivo", MENU_TOP_XS[0]),
                ("Casos", MENU_TOP_XS[1]),
                ("Sobre", MENU_TOP_XS[2]),
            ];
            for (text, lx) in &tops {
                menu_bar.push(Label::new(
                    fs,
                    text,
                    11.5,
                    menu_col,
                    lx + 10.0,
                    bar_y.max(4.0),
                ));
            }
        }

        // ── Título ──────────────────────────────────────────────────────────
        let mut title = Label::new_bold(fs, "NeuroScan", 28.0, col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = (h * 0.04).max(MENU_BAR_H + 8.0);

        // ── Subtítulo ────────────────────────────────────────────────────────
        let sub_text = if scan.case_id.is_empty() {
            "Visualizador Médico 3D  \u{00B7}  Arraste para girar  \u{00B7}  Scroll para zoom  \u{00B7}  I para painel  \u{00B7}  \u{2190}\u{2192} para navegar".to_string()
        } else {
            format!(
                "{}  \u{00B7}  {}  \u{00B7}  {}  \u{00B7}  I para painel  \u{00B7}  \u{2190}\u{2192} para navegar",
                scan.case_id, scan.dataset, scan.modalities
            )
        };
        let mut sub = Label::new(fs, &sub_text, 12.0, Color::rgb(148, 163, 184), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 3.0;

        always.push(title);
        always.push(sub);

        // ── Callouts ─────────────────────────────────────────────────────────
        let y_ct = h * 0.14;
        let box_w = (w * 0.175).clamp(190.0, 240.0);
        let _box_h = 92.0_f32;
        let pad = 12.0_f32;

        let col_micro = Color::rgb(148, 164, 182);

        // ET — canto superior esquerdo
        {
            let ex = 24.0;
            let ey = y_ct;
            let vol = if scan.et_volume_ml > 0.0 {
                format!("{:.1} mL", scan.et_volume_ml)
            } else {
                String::new()
            };
            always.push(Label::new(
                fs,
                "\u{25CF}  ET  \u{00B7}  Enhancing Tumor",
                10.5,
                rgb_f(ET_COLOR),
                ex + pad,
                ey + 20.0,
            ));
            if !vol.is_empty() {
                always.push(Label::new_bold(
                    fs,
                    &vol,
                    13.0,
                    Color::WHITE,
                    ex + pad,
                    ey + 38.0,
                ));
            }
            always.push(Label::new(
                fs,
                "Realce pós-contraste · barreira comprometida",
                8.8,
                col_micro,
                ex + pad,
                ey + 62.0,
            ));
        }

        // SNFH — labels em vec separado; posição definida por update_snfh_label_positions()
        {
            let vol_str = if scan.snfh_volume_ml > 0.0 {
                format!("{:.1} mL", scan.snfh_volume_ml)
            } else {
                String::new()
            };
            snfh.push(Label::new(
                fs,
                "\u{25CF}  SNFH  \u{00B7}  Peritumoral Edema",
                10.5,
                rgb_f(SNFH_COLOR),
                0.0,
                0.0,
            ));
            snfh.push(Label::new_bold(fs, &vol_str, 13.0, Color::WHITE, 0.0, 0.0));
            snfh.push(Label::new(
                fs,
                "Edema e infiltração peritumoral",
                8.8,
                col_micro,
                0.0,
                0.0,
            ));
        }

        // NETC — centro inferior
        {
            let nx = (w / 2.0 - box_w / 2.0).max(24.0);
            let ny = (h - 185.0).max(y_ct + 92.0 + 20.0);
            let vol = if scan.netc_volume_ml > 0.0 {
                format!("{:.1} mL", scan.netc_volume_ml)
            } else {
                String::new()
            };
            always.push(Label::new(
                fs,
                "\u{25CF}  NETC  \u{00B7}  Necrotic Core",
                10.5,
                rgb_f(NETC_COLOR),
                nx + pad,
                ny + 20.0,
            ));
            if !vol.is_empty() {
                always.push(Label::new_bold(
                    fs,
                    &vol,
                    13.0,
                    Color::WHITE,
                    nx + pad,
                    ny + 38.0,
                ));
            }
            always.push(Label::new(
                fs,
                "Núcleo necrótico hipóxico · centro da lesão",
                8.8,
                col_micro,
                nx + pad,
                ny + 62.0,
            ));
        }

        // ── Indicador de caso (centro inferior) ──────────────────────────────
        {
            let idx = current_case + 1;
            let n = TOP_CASES.len();
            let text = format!("\u{2190}   Caso {}  /  {}   \u{2192}", idx, n);
            let mut nav = Label::new(fs, &text, 11.0, Color::rgb(100, 116, 139), 0.0, 0.0);
            nav.x = (w - nav.measured_width()) / 2.0;
            nav.y = h - 22.0;
            always.push(nav);
        }

        // ── Marca d'água "NeuroScan" ─────────────────────────────────────────
        {
            let mut wm = Label::new_bold(
                fs,
                "NeuroScan",
                96.0,
                Color::rgba(210, 222, 238, 11),
                0.0,
                0.0,
            );
            wm.x = (w - wm.measured_width()) / 2.0;
            wm.y = (h - wm.line_height()) / 2.0 + 20.0;
            always.push(wm);
        }

        // ── Rodapé técnico (canto inferior direito) ───────────────────────────
        {
            let footer = format!(
                "NeuroScan AI  \u{00B7}  v{}  \u{00B7}  nnUNet 2D  \u{00B7}  BraTS 2021  \u{00B7}  Dice 0.865",
                env!("CARGO_PKG_VERSION")
            );
            let mut ft = Label::new(fs, &footer, 9.0, col_section(), 0.0, 0.0);
            ft.x = (w - ft.measured_width() - 18.0).max(0.0);
            ft.y = h - 16.0;
            always.push(ft);
        }

        // ── Legenda inferior esquerda ─────────────────────────────────────────
        let leg_font = 11.0;
        let leg_gap = leg_font * 1.8;
        let leg_items = [
            (ET_COLOR, "ET", scan.et_volume_ml),
            (SNFH_COLOR, "SNFH", scan.snfh_volume_ml),
            (NETC_COLOR, "NETC", scan.netc_volume_ml),
        ];
        let n_leg = leg_items.len() as f32;
        let leg_y = h - n_leg * leg_gap - 30.0;
        for (i, (rgb, name, vol)) in leg_items.iter().enumerate() {
            let text = if *vol > 0.0 {
                format!("\u{25CF}  {}  {:.1} mL", name, vol)
            } else {
                format!("\u{25CF}  {}", name)
            };
            always.push(Label::new(
                fs,
                &text,
                leg_font,
                rgb_f(*rgb),
                28.0,
                leg_y + i as f32 * leg_gap,
            ));
        }

        // ── Painel clínico (toggle I) ─────────────────────────────────────────
        let px = (w - 268.0).max(w * 0.72);
        let pl = px + 14.0;
        let mut py = h * 0.14;

        macro_rules! section {
            ($text:expr) => {{
                panel.push(Label::new(fs, $text, 9.5, col_section(), pl, py));
                py += 18.0;
            }};
        }
        macro_rules! kv {
            ($key:expr, $val:expr) => {{
                panel.push(Label::new(fs, $key, 10.0, col_dim(), pl, py));
                panel.push(Label::new(fs, $val, 11.0, col_value(), pl + 90.0, py));
                py += 16.0;
            }};
        }
        macro_rules! line_text {
            ($text:expr, $col:expr, $size:expr) => {{
                panel.push(Label::new(fs, $text, $size, $col, pl, py));
                py += $size * 1.5;
            }};
        }

        // Silence unused warning when show_panel is false (macros still compile the branches)
        let _ = show_panel;

        let case_val = if scan.case_id.is_empty() {
            "—".to_string()
        } else {
            scan.case_id.clone()
        };
        let dataset_val = if scan.dataset.is_empty() {
            "—".to_string()
        } else {
            scan.dataset.clone()
        };
        let nav_val = format!("{} / {}", current_case + 1, TOP_CASES.len());

        section!("ANÁLISE VOLUMÉTRICA");
        kv!("Caso", &case_val);
        kv!("Navegação", &nav_val);
        kv!("Protocolo", &dataset_val);
        if !scan.modalities.is_empty() {
            panel.push(Label::new(fs, "Modalidades", 10.0, col_dim(), pl, py));
            panel.push(Label::new(
                fs,
                "FLAIR · T1w",
                11.0,
                col_value(),
                pl + 90.0,
                py,
            ));
            py += 14.0;
            panel.push(Label::new(
                fs,
                "T1ce · T2w",
                11.0,
                col_value(),
                pl + 90.0,
                py,
            ));
            py += 16.0;
        }
        kv!("Método", "nnUNet 2D Slice");
        kv!("Acurácia", "Dice  0.865");
        py += 8.0;

        section!("CLASSIFICAÇÃO TUMORAL");
        line_text!("WHO 2021  ·  Grau IV", col_value(), 11.0);
        line_text!("Glioblastoma Multiforme", col_header(), 11.5);
        py += 6.0;

        section!("RISCO CLÍNICO");
        kv!("Grau WHO", "IV  —  Alto Risco");
        kv!("IDH Status", "Wild-type (wt)");
        kv!("MGMT Metil.", "A investigar");
        kv!("Ki-67", "> 30%");
        kv!("Critérios", "RANO 2010");
        kv!("Sobrevida med.", "14 – 16 meses");
        py += 3.0;
        panel.push(Label::new(
            fs,
            "* Dados pop. BraTS 2021. Não substitui laudo.",
            8.5,
            col_section(),
            pl,
            py,
        ));
        py += 14.0;
        py += 8.0;

        section!("VOLUMES SEGMENTADOS");
        let et_s = format!("{:.1} mL", scan.et_volume_ml);
        let snfh_s = format!("{:.1} mL", scan.snfh_volume_ml);
        let netc_s = format!("{:.1} mL", scan.netc_volume_ml);
        let tot_s = format!("{:.1} mL", scan.total_volume_ml);
        panel.push(Label::new(
            fs,
            "\u{25CF}  ET    Realce tumoral",
            10.5,
            rgb_f(ET_COLOR),
            pl,
            py,
        ));
        panel.push(Label::new(fs, &et_s, 11.5, Color::WHITE, pl + 168.0, py));
        py += 15.0;
        panel.push(Label::new(
            fs,
            "\u{25CF}  SNFH  Edema peritumoral",
            10.5,
            rgb_f(SNFH_COLOR),
            pl,
            py,
        ));
        panel.push(Label::new(fs, &snfh_s, 11.5, Color::WHITE, pl + 168.0, py));
        py += 15.0;
        panel.push(Label::new(
            fs,
            "\u{25CF}  NETC  Núcleo necrótico",
            10.5,
            rgb_f(NETC_COLOR),
            pl,
            py,
        ));
        panel.push(Label::new(fs, &netc_s, 11.5, Color::WHITE, pl + 168.0, py));
        py += 18.0;
        panel.push(Label::new(fs, "Total", 10.0, col_dim(), pl + 120.0, py));
        panel.push(Label::new_bold(
            fs,
            &tot_s,
            13.0,
            Color::WHITE,
            pl + 168.0,
            py,
        ));
        py += 22.0;

        section!("METODOLOGIA");
        kv!("Segmentação", "nnUNet 2D Slice");
        kv!("Treinamento", "484 casos BraTS 2021");
        kv!("Acurácia", "Dice 0.865");
        kv!("Modalidades", "FLAIR/T1w/T1ce/T2w");
        kv!("Resolução", "1mm isotrópico");
        kv!("Referência", "WHO CNS 2021");
        py += 4.0;
        panel.push(Label::new(
            fs,
            "Inferência slice-a-slice com 4",
            9.5,
            col_dim(),
            pl,
            py,
        ));
        py += 13.0;
        panel.push(Label::new(
            fs,
            "canais MRI simultaneos (ONNX).",
            9.5,
            col_dim(),
            pl,
            py,
        ));
        py += 13.0;
        py += 6.0;
        panel.push(Label::new(
            fs,
            format!("NeuroScan AI  v{}", env!("CARGO_PKG_VERSION")).as_str(),
            9.0,
            col_section(),
            pl,
            py,
        ));

        self.labels_always = always;
        self.labels_panel = panel;
        self.labels_snfh = snfh;
        self.labels_menu_bar = menu_bar;
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
        let bg = [0.04, 0.06, 0.11, 0.92_f32];
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
            b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, 0.72], w, h);
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
            b.rect(0.0, 0.0, w, h, [0.03, 0.04, 0.08, alpha], w, h);

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

        b
    }

    /// Constrói o batch de overlay do menu bar (Pass 2, acima do texto de cena).
    pub(crate) fn build_menu_overlay(&self, w: f32, h: f32) -> Prim2DBatch {
        use neuroscan_core::{MENU_DROP_W, MENU_ITEM_H, MENU_SEP_H, MENU_TOP_WS};

        let mut b = Prim2DBatch::new();

        b.rect(0.0, 0.0, w, MENU_BAR_H, [0.04, 0.06, 0.12, 0.97], w, h);
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

// Suppress unused import warning for SPLASH_FADEOUT_DURATION which is only used
// transitively via the public constant re-export from state.
const _: f32 = SPLASH_FADEOUT_DURATION;
const _: usize = TUMOR_COUNT;
