use crate::camera::OrbitalCamera;
use crate::mesh::Mesh;
use neuroscan_core::{CASES_DIR, ScanMeta, TOP_CASES};
use std::sync::{Arc, mpsc};
use std::time::Instant;
use wgpu::Device;
use winit::window::Window;

use crate::renderer::GpuState;
use crate::ui::Label;

// --- Cores RGB linear ---
pub(crate) const LH_COLOR: [f32; 3] = [0.90, 0.72, 0.70];
pub(crate) const RH_COLOR: [f32; 3] = [0.88, 0.70, 0.68];
pub(crate) const CEREB_COLOR: [f32; 3] = [0.82, 0.62, 0.60];
pub(crate) const INNER_COLOR: [f32; 3] = [0.78, 0.65, 0.72];
pub(crate) const VENTR_COLOR: [f32; 3] = [0.55, 0.72, 0.85];
pub(crate) const ET_COLOR: [f32; 3] = [0.95, 0.18, 0.12];
pub(crate) const SNFH_COLOR: [f32; 3] = [0.95, 0.78, 0.05];
pub(crate) const NETC_COLOR: [f32; 3] = [0.25, 0.50, 0.98];

pub(crate) const LH_ALPHA: f32 = 0.30;
pub(crate) const RH_ALPHA: f32 = 0.30;
pub(crate) const CEREB_ALPHA: f32 = 0.55;
pub(crate) const INNER_ALPHA: f32 = 0.80;
pub(crate) const VENTR_ALPHA: f32 = 0.45;

// --- Camera ---
pub(crate) const MOUSE_SENSITIVITY: f32 = 0.005;
pub(crate) const ZOOM_SENSITIVITY: f32 = 0.3;
pub(crate) const ZOOM_MIN: f32 = 1.5;
pub(crate) const ZOOM_MAX: f32 = 20.0;
pub(crate) const PITCH_LIMIT: f32 = 1.5;
pub(crate) const AUTO_ROTATE_IDLE_S: f32 = 5.0;
pub(crate) const AUTO_ROTATE_SPEED: f32 = 0.008;

// --- Animações ---
pub(crate) const TRANSITION_DURATION: f32 = 0.45;
pub(crate) const PULSE_FREQ: f32 = 1.2;
pub(crate) const SPLASH_FADEOUT_DURATION: f32 = 0.70;
/// Duração do fade inferência → visualização 3D (segundos).
pub(crate) const INFER_FADE_DURATION: f32 = 0.80;

// --- Janela ---
pub(crate) const WINDOW_WIDTH: f64 = 1280.0;
pub(crate) const WINDOW_HEIGHT: f64 = 720.0;
pub(crate) const ICON_BYTES: &[u8] = include_bytes!("../../assets/icon_256x256.png");

/// Meshes permanentes do cérebro (compartilhadas entre todos os casos)
pub(crate) const BRAIN_DEFS: &[(&str, [f32; 3], f32)] = &[
    (
        concat!("assets/models/premium/", "Ventricles.obj"),
        VENTR_COLOR,
        VENTR_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Thalmus_and_Optic_Tract.obj"),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Corpus_Callosum.obj"),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!(
            "assets/models/premium/",
            "Hippocampus_and_Indusium_Griseum.obj"
        ),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Putamen_and_Amygdala.obj"),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Globus_Pallidus_Externus.obj"),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Globus_Pallidus_Internus.obj"),
        INNER_COLOR,
        INNER_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Cerebellum.obj"),
        CEREB_COLOR,
        CEREB_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Medulla_and_Pons.obj"),
        CEREB_COLOR,
        CEREB_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Left_Cerebral_Hemisphere.obj"),
        LH_COLOR,
        LH_ALPHA,
    ),
    (
        concat!("assets/models/premium/", "Right_Cerebral_Hemisphere.obj"),
        RH_COLOR,
        RH_ALPHA,
    ),
];

pub(crate) const TUMOR_COUNT: usize = 3;

/// Modo de visualizacao do cerebro — alternado via F2/F3.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum BrainViewMode {
    /// Cerebro transparente + tumores visiveis (default, analise clinica).
    Transparent,
    /// Sem cerebro, apenas tumores (analise cirurgica).
    TumorsOnly,
    /// Cerebro opaco/realista (apresentacao anatomica).
    Opaque,
}

pub(crate) struct LoadedMesh {
    pub(crate) mesh: Mesh,
    pub(crate) tint: [f32; 3],
    pub(crate) alpha: f32,
}

impl BrainViewMode {
    /// Retorna o alpha efetivo de um mesh dado o modo e sua posicao no array.
    ///
    /// Indices 0..TUMOR_COUNT sao tumores (sempre visiveis).
    /// Indices TUMOR_COUNT.. sao cerebro (controlados pelo modo).
    pub(crate) fn effective_alpha(self, index: usize, base_alpha: f32) -> Option<f32> {
        let is_brain = index >= TUMOR_COUNT;
        if !is_brain {
            // Tumores: sempre visiveis, alpha original
            return Some(base_alpha);
        }
        match self {
            BrainViewMode::Transparent => Some(base_alpha),
            BrainViewMode::TumorsOnly => None,   // oculto
            BrainViewMode::Opaque => Some(0.95), // quase opaco, leve translucidez
        }
    }
}

pub(crate) struct App {
    pub(crate) window: Option<Arc<Window>>,
    pub(crate) gpu: Option<GpuState>,
    /// Índices 0..TUMOR_COUNT = tumores do caso atual; o resto = cérebro permanente
    pub(crate) meshes: Vec<LoadedMesh>,
    pub(crate) camera: OrbitalCamera,
    pub(crate) centroids: [glam::Vec3; TUMOR_COUNT],
    pub(crate) labels_always: Vec<Label>,
    pub(crate) labels_panel: Vec<Label>,
    pub(crate) labels_snfh: Vec<Label>,
    pub(crate) labels_menu_bar: Vec<Label>,
    pub(crate) labels_menu: Vec<Label>,
    pub(crate) show_panel: bool,
    /// Modo de visualizacao do cerebro (F2/F3).
    pub(crate) brain_view: BrainViewMode,
    // Menu bar
    pub(crate) menu_open: i32,
    pub(crate) menu_hover_top: i32,
    pub(crate) menu_hover_item: i32,
    pub(crate) scan: ScanMeta,
    // Janela
    pub(crate) window_shown: bool,
    // Splash screen
    pub(crate) splash_done: bool,
    pub(crate) splash_t: f32,
    pub(crate) splash_fade: f32,
    pub(crate) splash_rx: Option<mpsc::Receiver<Vec<LoadedMesh>>>,
    pub(crate) splash_labels: Vec<Label>,
    // Tela inicial (home screen) — mostrada após splash, antes de qualquer inferência
    pub(crate) show_home: bool,
    pub(crate) home_labels: Vec<Label>,
    pub(crate) home_anim_t: f32,
    // Erro de inferencia — exibido na home screen
    pub(crate) python_env_error: Option<String>,
    // Inferência — pipeline nativo Rust (ort + nifti)
    pub(crate) dialog_rx: Option<mpsc::Receiver<std::path::PathBuf>>,
    pub(crate) infer_active: bool,
    pub(crate) infer_rx: Option<mpsc::Receiver<crate::app::infer::InferMsg>>,
    pub(crate) infer_progress: Option<crate::app::infer::InferProgress>,
    pub(crate) infer_labels: Vec<Label>,
    /// Fade de transição inferência → 3D (0.0 = tela infer, 1.0 = 3D completo).
    pub(crate) infer_fade: f32,
    // Navegação entre casos
    pub(crate) current_case: usize,
    // Animação de transição entre casos
    pub(crate) transition_phase: f32,
    pub(crate) transition_target: usize,
    pub(crate) loading_rx: Option<mpsc::Receiver<Vec<LoadedMesh>>>,
    // Microanimações
    pub(crate) last_interaction: Instant,
    pub(crate) pulse_t: f32,
    pub(crate) spinner_angle: f32,
    pub(crate) snfh_anim_t: f32,
    pub(crate) last_frame: Instant,
    pub(crate) mouse_pressed: bool,
    pub(crate) mouse_pos: Option<(f64, f64)>,
}

impl App {
    pub(crate) fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            meshes: Vec::new(),
            camera: OrbitalCamera::new(4.0),
            centroids: [glam::Vec3::ZERO; TUMOR_COUNT],
            labels_always: Vec::new(),
            labels_panel: Vec::new(),
            labels_snfh: Vec::new(),
            labels_menu_bar: Vec::new(),
            labels_menu: Vec::new(),
            show_panel: false,
            brain_view: BrainViewMode::Transparent,
            menu_open: -1,
            menu_hover_top: -1,
            menu_hover_item: -1,
            scan: ScanMeta::default(),
            window_shown: false,
            splash_done: false,
            splash_t: 0.0,
            splash_fade: 0.0,
            splash_rx: None,
            splash_labels: Vec::new(),
            show_home: true,
            home_labels: Vec::new(),
            home_anim_t: 0.0,
            python_env_error: None,
            dialog_rx: None,
            infer_active: false,
            infer_rx: None,
            infer_progress: None,
            infer_labels: Vec::new(),
            infer_fade: 0.0,
            current_case: 0,
            transition_phase: 0.0,
            transition_target: 0,
            loading_rx: None,
            last_interaction: Instant::now(),
            pulse_t: 0.0,
            spinner_angle: 0.0,
            snfh_anim_t: 0.0,
            last_frame: Instant::now(),
            mouse_pressed: false,
            mouse_pos: None,
        }
    }
}

/// Spawns a background thread to load all brain meshes (brain permanents + tumors for case_id).
/// Returns a receiver that yields `Vec<LoadedMesh>` when done.
/// `wgpu::Device` is internally reference-counted — cloning is cheap.
pub(crate) fn load_brain_meshes_bg(
    device: Device,
    case_id: &str,
) -> mpsc::Receiver<Vec<LoadedMesh>> {
    let case_id = case_id.to_string();
    let (tx, rx) = mpsc::channel::<Vec<LoadedMesh>>();
    std::thread::spawn(move || {
        let case_dir = format!("{}/{}", CASES_DIR, case_id);
        let tumor_defs = [
            (format!("{}/tumor_et.obj", case_dir), ET_COLOR, 1.0_f32),
            (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0_f32),
            (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0_f32),
        ];
        let mut all: Vec<LoadedMesh> = tumor_defs
            .iter()
            .filter_map(|(path, tint, alpha)| {
                crate::mesh::Mesh::from_obj(&device, path)
                    .ok()
                    .map(|m| LoadedMesh {
                        mesh: m,
                        tint: *tint,
                        alpha: *alpha,
                    })
            })
            .collect();
        for (path, tint, alpha) in BRAIN_DEFS {
            // Extrai filename do path (ex: "assets/models/premium/Ventricles.obj" -> "Ventricles.obj")
            let filename = std::path::Path::new(path)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or(path);
            let mesh_result = if let Some(file) = crate::embedded::premium_obj_bytes(filename) {
                crate::mesh::Mesh::from_obj_bytes(&device, &file.data)
            } else {
                // Fallback: disco (desenvolvimento)
                crate::mesh::Mesh::from_obj(&device, path)
            };
            if let Ok(m) = mesh_result {
                all.push(LoadedMesh {
                    mesh: m,
                    tint: *tint,
                    alpha: *alpha,
                });
            }
        }
        let _ = tx.send(all);
    });
    rx
}

/// Spawns a background thread to load only the tumor meshes for `case_id`.
pub(crate) fn load_tumor_meshes_bg(
    device: Device,
    case_id: &str,
) -> mpsc::Receiver<Vec<LoadedMesh>> {
    let case_id = case_id.to_string();
    let (tx, rx) = mpsc::channel::<Vec<LoadedMesh>>();
    std::thread::spawn(move || {
        let case_dir = format!("{}/{}", CASES_DIR, case_id);
        let defs = [
            (format!("{}/tumor_et.obj", case_dir), ET_COLOR, 1.0_f32),
            (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0_f32),
            (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0_f32),
        ];
        let meshes: Vec<LoadedMesh> = defs
            .iter()
            .filter_map(|(path, tint, alpha)| {
                crate::mesh::Mesh::from_obj(&device, path)
                    .ok()
                    .map(|m| LoadedMesh {
                        mesh: m,
                        tint: *tint,
                        alpha: *alpha,
                    })
            })
            .collect();
        let _ = tx.send(meshes);
    });
    rx
}

impl App {
    /// Inicia a transição animada para o caso no índice `target`.
    /// O carregamento das meshes acontece em thread de fundo — sem travar o render loop.
    pub(crate) fn navigate_case(&mut self, dir: i32) {
        if self.transition_phase > 0.0 {
            return;
        }
        let n = TOP_CASES.len();
        let next = ((self.current_case as i32 + dir).rem_euclid(n as i32)) as usize;
        self.transition_target = next;
        self.transition_phase = f32::EPSILON;
        self.spinner_angle = 0.0;
        self.last_interaction = Instant::now();

        // wgpu::Device is internally Arc<..> — clone is cheap
        let device: Device = self.gpu.as_ref().unwrap().device.clone();
        self.loading_rx = Some(load_tumor_meshes_bg(device, TOP_CASES[next]));
    }

    #[allow(dead_code)]
    pub(crate) fn load_tumor_meshes(&mut self, case_id: &str) {
        let case_dir = format!("{}/{}", CASES_DIR, case_id);
        let defs = [
            (format!("{}/tumor_et.obj", case_dir), ET_COLOR, 1.0f32),
            (format!("{}/tumor_snfh.obj", case_dir), SNFH_COLOR, 1.0f32),
            (format!("{}/tumor_netc.obj", case_dir), NETC_COLOR, 1.0f32),
        ];

        let new_meshes: Vec<LoadedMesh> = {
            let Some(gpu) = &self.gpu else { return };
            let mut result = Vec::new();
            for (path, tint, alpha) in &defs {
                match crate::mesh::Mesh::from_obj(&gpu.device, path) {
                    Ok(m) => result.push(LoadedMesh {
                        mesh: m,
                        tint: *tint,
                        alpha: *alpha,
                    }),
                    Err(e) => {
                        tracing::warn!(error = %e, path = %path, "mesh de tumor nao encontrada")
                    }
                }
            }
            result
        };

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

        self.scan = ScanMeta::load(&ScanMeta::case_path(case_id));
        tracing::info!(
            case = case_id,
            total_ml = self.scan.total_volume_ml,
            "caso carregado"
        );
    }
}
