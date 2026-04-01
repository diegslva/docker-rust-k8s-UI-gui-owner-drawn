use neuroscan_core::MENU_BAR_H;
use neuroscan_core::MENU_TOP_XS;
use neuroscan_core::TOP_CASES;
use winit::dpi::PhysicalSize;

use crate::app::App;
use crate::app::state::{ET_COLOR, NETC_COLOR, SNFH_COLOR};
use crate::ui::{Color, Label};

use super::{col_dim, col_header, col_section, col_subtitle, col_value, rgb_f};

impl App {
    pub(crate) fn build_splash_labels(&mut self, size: PhysicalSize<u32>) {
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let cy = h / 2.0 + 28.0;
        let fs = gpu.font_system_mut();

        let mut title = Label::new_bold(fs, "NeuroScan", 50.0, col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = cy - 122.0;

        let mut sub = Label::new(fs, "Visualizador Médico 3D", 15.5, col_subtitle(), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = title.y + title.line_height() + 5.0;

        // Versao discreta no canto inferior
        let ver_text = format!("v{}", env!("CARGO_PKG_VERSION"));
        let mut ver = Label::new(fs, &ver_text, 9.0, Color::rgb(60, 75, 95), 0.0, 0.0);
        ver.x = (w - ver.measured_width()) / 2.0;
        ver.y = h - 28.0;

        self.splash_labels = vec![title, sub, ver];
    }

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
        let mut sub = Label::new(fs, &sub_text, 12.0, col_subtitle(), 0.0, 0.0);
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

        // ── Hints permanentes (canto inferior esquerdo) ──
        {
            let hint_col = Color::rgb(80, 115, 155);
            let back = Label::new(
                fs,
                "Esc  \u{2190}  Tela inicial",
                10.5,
                hint_col,
                28.0,
                h - 20.0,
            );
            always.push(back);
            let help_hint = Label::new(fs, "H  Atalhos", 10.5, hint_col, 180.0, h - 20.0);
            always.push(help_hint);
            let measure_hint = Label::new(fs, "M  Medir", 10.5, hint_col, 280.0, h - 20.0);
            always.push(measure_hint);
            let gimbal_hint = Label::new(fs, "G  Orientacao", 10.5, hint_col, 360.0, h - 20.0);
            always.push(gimbal_hint);
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
                "NeuroScan AI  \u{00B7}  v{}  \u{00B7}  nnUNet 2D  \u{00B7}  BraTS 2021+2023  \u{00B7}  1.735 casos",
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
            "* Dados pop. BraTS 2021+2023. Nao substitui laudo.",
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
        kv!("Treinamento", "1.735 casos BraTS 2021+2023");
        kv!("Acurácia", "Dice 0.822 (ET 0.924)");
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

        // --- Tooltip contextual (fade out automatico) ---
        if let Some(text) = &self.tooltip_text {
            if self.tooltip_timer > 0.0 {
                let alpha = (self.tooltip_timer.min(1.0) * 255.0) as u8;
                let mut tt =
                    Label::new_bold(fs, text, 12.0, Color::rgba(200, 222, 245, alpha), 0.0, 0.0);
                tt.x = (w - tt.measured_width()) / 2.0;
                tt.y = h - 55.0;
                always.push(tt);
            }
        }

        // --- Callout do slice plane ---
        if self.slice_visible && self.volume.is_some() {
            let plane_name = match self.slice_plane {
                crate::volume::SlicePlane::Axial => "Axial",
                crate::volume::SlicePlane::Coronal => "Coronal",
                crate::volume::SlicePlane::Sagittal => "Sagital",
            };
            let pct = (self.slice_position * 100.0).round() as u32;
            let slice_text = format!(
                "Corte {} -- {}%  |  Shift+scroll para mover  |  1/2/3 plano",
                plane_name, pct
            );
            let mut sl = Label::new(fs, &slice_text, 10.0, col_subtitle(), 0.0, 0.0);
            sl.x = (w - sl.measured_width()) / 2.0;
            sl.y = MENU_BAR_H + 72.0;
            always.push(sl);
        }

        // --- Labels N/S do gimbal ---
        if self.show_gimbal {
            let radius = 1.2_f32;
            let cam_u = self.camera.build_uniform(w as u32, h as u32);
            let north = glam::Vec3::new(0.0, radius * 1.12, 0.0);
            let south = glam::Vec3::new(0.0, -radius * 1.12, 0.0);
            if let Some((nx, ny)) =
                crate::app::projection::project_to_screen(north, &cam_u.mvp, w, h)
            {
                let lbl = Label::new_bold(
                    fs,
                    "N",
                    12.0,
                    Color::rgb(140, 190, 240),
                    nx - 5.0,
                    ny - 18.0,
                );
                always.push(lbl);
            }
            if let Some((sx, sy)) =
                crate::app::projection::project_to_screen(south, &cam_u.mvp, w, h)
            {
                let lbl =
                    Label::new_bold(fs, "S", 12.0, Color::rgb(110, 155, 200), sx - 5.0, sy + 6.0);
                always.push(lbl);
            }
        }

        // --- Callout de medicao (MESMO padrao que ET/SNFH/NETC) ---
        if self.measure_active {
            let measure_col = Color::rgb(255, 200, 100);
            let col_micro = Color::rgb(148, 164, 182);

            // Posicao: mesma logica que SNFH (canto direito), mas uma box abaixo
            let bh = 92.0_f32;
            let mx = w - box_w - 24.0;
            let my = y_ct + bh + 12.0 + bh + 12.0;

            if let (Some(a), Some(b)) = (&self.measure_point_a, &self.measure_point_b) {
                let scale = self.volume.as_ref().map_or(181.28_f32, |v| v.scale as f32);
                let up = self
                    .volume
                    .as_ref()
                    .map_or(2.0_f32, |v| v.upsample_factor as f32);
                let dist = crate::app::projection::distance_mm(a.world_pos, b.world_pos, scale, up);
                let dist_text = format!("{:.1} mL", dist);
                let vol_str = format!("{:.1} mm", dist);

                // Titulo (mesmo y + 20 que ET)
                always.push(Label::new(
                    fs,
                    "\u{25CF}  Medicao  \u{00B7}  Distancia",
                    10.5,
                    measure_col,
                    mx + pad,
                    my + 20.0,
                ));
                // Valor grande (mesmo y + 38 que ET)
                always.push(Label::new_bold(
                    fs,
                    &vol_str,
                    13.0,
                    Color::WHITE,
                    mx + pad,
                    my + 38.0,
                ));
                // Descricao (mesmo y + 62 que ET)
                always.push(Label::new(
                    fs,
                    "Clique para mover B  \u{00B7}  M para limpar",
                    8.8,
                    col_micro,
                    mx + pad,
                    my + 62.0,
                ));

                // (distancia ja mostrada no callout fixo — sem label duplicado no 3D)
            } else if self.measure_point_a.is_some() {
                always.push(Label::new(
                    fs,
                    "\u{25CF}  Medicao",
                    10.5,
                    measure_col,
                    mx + pad,
                    my + 20.0,
                ));
                always.push(Label::new(
                    fs,
                    "Clique no segundo ponto",
                    10.0,
                    col_micro,
                    mx + pad,
                    my + 40.0,
                ));
            } else {
                always.push(Label::new(
                    fs,
                    "\u{25CF}  Medicao",
                    10.5,
                    measure_col,
                    mx + pad,
                    my + 20.0,
                ));
                always.push(Label::new(
                    fs,
                    "Clique no primeiro ponto",
                    10.0,
                    col_micro,
                    mx + pad,
                    my + 40.0,
                ));
            }
        }

        // --- Help overlay (H) com fade-in/out ---
        if self.help_anim_t > 0.01 {
            let ha = self.help_anim_t;
            let help_items = [
                ("H", "Mostrar/ocultar esta ajuda"),
                ("I", "Painel clinico (dados volumetricos)"),
                ("O", "Abrir volume NIfTI"),
                ("F2", "Cerebro transparente / apenas tumores"),
                ("F3", "Cerebro opaco (anatomia realista)"),
                ("F11", "Tela cheia"),
                ("4", "Corte MRI (plano de ressonancia)"),
                ("1/2/3", "Plano axial / coronal / sagital"),
                ("Shift+Scroll", "Mover plano de corte"),
                ("M", "Medicao (distancia entre dois pontos)"),
                ("G", "Orientacao 3D (aneis N/S)"),
                ("Esc", "Voltar a tela inicial"),
                ("Setas", "Navegar entre casos clinicos"),
            ];
            let box_w = 380.0_f32;
            let box_x = (w - box_w) / 2.0;
            let mut hy = h * 0.18;

            let alpha_u8 = (ha * 255.0) as u8;
            let mut title = Label::new_bold(
                fs,
                "Atalhos do NeuroScan",
                14.0,
                Color::rgba(200, 222, 245, alpha_u8),
                0.0,
                0.0,
            );
            title.x = (w - title.measured_width()) / 2.0;
            title.y = hy;
            always.push(title);
            hy += 28.0;

            for (key, desc) in &help_items {
                let key_text = format!("  {}  ", key);
                let kl = Label::new_bold(
                    fs,
                    &key_text,
                    11.0,
                    Color::rgba(120, 180, 240, alpha_u8),
                    box_x,
                    hy,
                );
                always.push(kl);
                let dl = Label::new(
                    fs,
                    desc,
                    11.0,
                    Color::rgba(160, 192, 220, alpha_u8),
                    box_x + 120.0,
                    hy,
                );
                always.push(dl);
                hy += 22.0;
            }
        }

        self.labels_always = always;
        self.labels_panel = panel;
        self.labels_snfh = snfh;
        self.labels_menu_bar = menu_bar;
    }

    /// Constroi os labels da tela inicial (home screen).
    ///
    /// Mostrada apos a splash, antes de qualquer inferencia. Apresenta o
    /// NeuroScan como produto: branding, capacidades, call-to-action.
    pub(crate) fn build_home_labels(&mut self, size: PhysicalSize<u32>) {
        let anim_t = self.home_anim_t;
        let python_err = self.python_env_error.clone();
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let fs = gpu.font_system_mut();

        let mut labels: Vec<Label> = Vec::new();

        // Titulo principal
        let mut title = Label::new_bold(fs, "NeuroScan AI", 42.0, col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.12;
        labels.push(title);

        // Subtitulo descritivo
        let mut sub = Label::new(
            fs,
            "Segmentação Cerebral Inteligente",
            15.0,
            col_subtitle(),
            0.0,
            0.0,
        );
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = h * 0.12 + 54.0;
        labels.push(sub);

        // Capacidades (abaixo da animacao cerebral)
        let features: &[(&str, [f32; 3])] = &[
            ("Inferencia nnUNet 2D slice-a-slice", [0.65, 0.75, 0.88]),
            (
                "4 canais MRI: FLAIR  ·  T1w  ·  T1ce  ·  T2w",
                [0.65, 0.75, 0.88],
            ),
            (
                "Segmentacao tumoral: ET  ·  SNFH  ·  NETC",
                [0.65, 0.75, 0.88],
            ),
            (
                "Visualizacao 3D interativa com dados reais",
                [0.65, 0.75, 0.88],
            ),
        ];
        let feat_y = h * 0.62;
        for (i, (text, col)) in features.iter().enumerate() {
            let feat_col = Color::rgb(
                (col[0] * 255.0) as u8,
                (col[1] * 255.0) as u8,
                (col[2] * 255.0) as u8,
            );
            let mut lbl = Label::new(fs, text, 11.5, feat_col, 0.0, 0.0);
            lbl.x = (w - lbl.measured_width()) / 2.0;
            lbl.y = feat_y + i as f32 * 22.0;
            labels.push(lbl);
        }

        // Separador sutil abaixo das features
        // (renderizado via primitivas, nao label)

        // Call-to-action (breathing alpha)
        let cta_y = feat_y + features.len() as f32 * 22.0 + 36.0;
        if let Some(err) = &python_err {
            // Erro de ambiente: mostra mensagem de aviso em âmbar
            let mut err_lbl = Label::new_bold(fs, err, 11.5, Color::rgb(240, 180, 60), 0.0, 0.0);
            err_lbl.x = (w - err_lbl.measured_width()) / 2.0;
            err_lbl.y = cta_y;
            labels.push(err_lbl);
        } else {
            // Breathing: alpha oscila suavemente entre 160 e 255
            let breath_alpha = (180.0 + 50.0 * (anim_t * 1.5).sin()) as u8;
            let mut cta = Label::new_bold(
                fs,
                "Arquivo  >  Abrir Volume NIfTI para iniciar",
                13.0,
                Color::rgba(120, 180, 240, breath_alpha),
                0.0,
                0.0,
            );
            cta.x = (w - cta.measured_width()) / 2.0;
            cta.y = cta_y;
            labels.push(cta);
        }

        // Footer
        let footer = format!(
            "NeuroScan AI  v{}  ·  Diego L. Silva  ·  github.com/diegslva",
            env!("CARGO_PKG_VERSION")
        );
        let mut ft = Label::new(fs, &footer, 9.0, col_section(), 0.0, 0.0);
        ft.x = (w - ft.measured_width()) / 2.0;
        ft.y = h - 22.0;
        labels.push(ft);

        self.home_labels = labels;
    }

    /// Constroi os labels da tela de inferencia.
    ///
    /// Chamado a cada frame durante inferencia ativa — o contador de fatias
    /// muda rapido e precisa ser refletido nos labels.
    pub(crate) fn build_infer_labels(&mut self, size: PhysicalSize<u32>) {
        let progress = self.infer_progress.clone().unwrap_or_default();
        let Some(gpu) = &mut self.gpu else { return };
        let w = size.width as f32;
        let h = size.height as f32;
        let fs = gpu.font_system_mut();

        let mut labels: Vec<Label> = Vec::new();

        // Titulo centralizado
        let mut title = Label::new_bold(fs, "NeuroScan AI", 36.0, col_header(), 0.0, 0.0);
        title.x = (w - title.measured_width()) / 2.0;
        title.y = h * 0.15;
        labels.push(title);

        // Subtitulo
        let mut sub = Label::new(fs, "Processando volume MRI", 14.0, col_subtitle(), 0.0, 0.0);
        sub.x = (w - sub.measured_width()) / 2.0;
        sub.y = h * 0.15 + 48.0;
        labels.push(sub);

        // Fase atual
        use crate::app::infer::InferPhase;
        // Fracao de progresso (usada para percentual e barra)
        let frac = if progress.total_slices > 0 {
            match &progress.phase {
                InferPhase::PythonSetup => 0.01_f32,
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
        let pct = (frac * 100.0).round() as u32;

        let phase_text = match &progress.phase {
            InferPhase::PythonSetup => "Preparando ambiente de inferencia...".to_string(),
            InferPhase::Preprocessing => "Carregando e normalizando volume...".to_string(),
            InferPhase::Slicing => {
                if progress.total_slices > 0 {
                    format!(
                        "Inferindo fatia {} de {} ({}%)",
                        progress.current_slice, progress.total_slices, pct
                    )
                } else {
                    "Iniciando inferencia...".to_string()
                }
            }
            InferPhase::MarchingCubes => "Extraindo superficies 3D (Marching Cubes)...".to_string(),
            InferPhase::Done => "Concluido".to_string(),
            InferPhase::Error(e) => format!("Erro: {}", e),
        };

        let cy = h * 0.70;
        let mut phase_lbl = Label::new_bold(fs, &phase_text, 14.0, col_header(), 0.0, 0.0);
        phase_lbl.x = (w - phase_lbl.measured_width()) / 2.0;
        phase_lbl.y = cy;
        labels.push(phase_lbl);

        // Percentual grande acima da barra
        let pct_text = format!("{}%", pct);
        let mut pct_lbl = Label::new_bold(fs, &pct_text, 22.0, Color::rgb(255, 255, 255), 0.0, 0.0);
        pct_lbl.x = (w - pct_lbl.measured_width()) / 2.0;
        pct_lbl.y = h * 0.78;
        labels.push(pct_lbl);

        // Tempo decorrido + ETA estimado
        let elapsed_secs = progress.elapsed_secs;
        let eta_text = if progress.current_slice > 5 && progress.total_slices > 0 {
            let remaining = progress.total_slices.saturating_sub(progress.current_slice);
            let secs_per_slice = elapsed_secs / progress.current_slice as f32;
            let eta = (remaining as f32 * secs_per_slice).round() as u32;
            if eta > 0 {
                format!("{:.0}s  --  ~{}s restantes", elapsed_secs, eta)
            } else {
                format!("{:.0}s", elapsed_secs)
            }
        } else {
            format!("{:.0}s", elapsed_secs)
        };
        let mut elapsed_lbl = Label::new(fs, &eta_text, 11.0, col_dim(), 0.0, 0.0);
        elapsed_lbl.x = (w - elapsed_lbl.measured_width()) / 2.0;
        elapsed_lbl.y = cy + 24.0;
        labels.push(elapsed_lbl);

        // Volumes parciais — linha horizontal abaixo da barra de progresso
        let vol_y = h * 0.84 + 22.0;
        let vol_bright = (progress.anim_t * 3.0).sin().abs() * 0.10;
        let vol_items = [
            ("ET", progress.et_volume_ml, ET_COLOR),
            ("SNFH", progress.snfh_volume_ml, SNFH_COLOR),
            ("NETC", progress.netc_volume_ml, NETC_COLOR),
        ];
        // Construir texto inline: "● ET 2.3 mL   ● SNFH 8.7 mL   ● NETC 1.5 mL"
        let mut parts: Vec<(String, Color)> = Vec::new();
        for (name, ml, color) in &vol_items {
            if *ml > 0.0 {
                let bright_col = Color::rgb(
                    ((color[0] + vol_bright).min(1.0) * 255.0) as u8,
                    ((color[1] + vol_bright).min(1.0) * 255.0) as u8,
                    ((color[2] + vol_bright).min(1.0) * 255.0) as u8,
                );
                parts.push((format!("\u{25CF} {} {:.1} mL", name, ml), bright_col));
            }
        }
        if !parts.is_empty() {
            // Renderizar cada parte lado a lado com spacing
            let spacing = 28.0_f32;
            let total_w: f32 = parts
                .iter()
                .map(|(t, _)| {
                    let lbl = Label::new(fs, t, 11.0, Color::rgb(255, 255, 255), 0.0, 0.0);
                    lbl.measured_width()
                })
                .sum::<f32>()
                + spacing * (parts.len() as f32 - 1.0);
            let mut x = (w - total_w) / 2.0;
            for (text, col) in &parts {
                let mut lbl = Label::new(fs, text, 11.0, *col, 0.0, 0.0);
                lbl.x = x;
                lbl.y = vol_y;
                x += lbl.measured_width() + spacing;
                labels.push(lbl);
            }
        }

        self.infer_labels = labels;
    }
}
