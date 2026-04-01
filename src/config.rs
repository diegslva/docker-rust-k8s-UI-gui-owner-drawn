//! Persistencia de configuracoes do NeuroScan.
//!
//! Salva/carrega preferencias do usuario em `neuroscan_data/config.json`.
//! Carregado no startup, salvo ao fechar ou ao mudar preferencia.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

/// Configuracoes persistidas do usuario.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    /// Usuario do ultimo login (se "Lembrar-me" ativo).
    #[serde(default)]
    pub remembered_user: Option<String>,

    /// Ultimo caso visualizado (indice no TOP_CASES).
    #[serde(default)]
    pub last_case_index: usize,

    /// Painel clinico visivel (I).
    #[serde(default)]
    pub show_panel: bool,

    /// Gimbal de orientacao visivel (G).
    #[serde(default)]
    pub show_gimbal: bool,

    /// Modo de visualizacao do cerebro: "transparent", "tumors_only", "opaque".
    #[serde(default = "default_brain_view")]
    pub brain_view: String,

    /// Slice plane visivel (4).
    #[serde(default)]
    pub slice_visible: bool,

    /// Plano do slice: "axial", "coronal", "sagittal".
    #[serde(default = "default_slice_plane")]
    pub slice_plane: String,

    /// Posicao do slice (0.0..1.0).
    #[serde(default = "default_slice_position")]
    pub slice_position: f32,

    /// Distancia da camera (zoom).
    #[serde(default = "default_camera_distance")]
    pub camera_distance: f32,
}

fn default_brain_view() -> String {
    "transparent".to_string()
}
fn default_slice_plane() -> String {
    "axial".to_string()
}
fn default_slice_position() -> f32 {
    0.5
}
fn default_camera_distance() -> f32 {
    4.0
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            remembered_user: None,
            last_case_index: 0,
            show_panel: false,
            show_gimbal: false,
            brain_view: default_brain_view(),
            slice_visible: false,
            slice_plane: default_slice_plane(),
            slice_position: default_slice_position(),
            camera_distance: default_camera_distance(),
        }
    }
}

impl AppConfig {
    /// Path do config.json (ao lado do executavel).
    fn config_path() -> PathBuf {
        std::env::current_exe()
            .ok()
            .and_then(|p| {
                p.parent()
                    .map(|d| d.join("neuroscan_data").join("config.json"))
            })
            .unwrap_or_else(|| PathBuf::from("neuroscan_data/config.json"))
    }

    /// Carrega config do disco. Retorna default se nao existe ou falha.
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<AppConfig>(&text) {
                Ok(cfg) => {
                    info!(path = %path.display(), "config carregado");
                    cfg
                }
                Err(e) => {
                    warn!(error = %e, "config.json invalido, usando default");
                    Self::default()
                }
            },
            Err(_) => {
                info!("config.json nao encontrado, usando default");
                Self::default()
            }
        }
    }

    /// Salva config em disco. Cria diretorio se necessario.
    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, &json) {
                    warn!(error = %e, "falha ao salvar config.json");
                } else {
                    info!(path = %path.display(), "config salvo");
                }
            }
            Err(e) => {
                warn!(error = %e, "falha ao serializar config");
            }
        }
    }
}
