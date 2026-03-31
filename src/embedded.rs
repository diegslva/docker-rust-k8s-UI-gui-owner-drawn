//! Assets embarcados no binario via rust-embed.
//!
//! Extrai para disco os assets que Python precisa (ONNX model, brain_meta, script).
//! OBJs premium sao carregados direto da memoria sem extrair.

use rust_embed::Embed;
use std::path::{Path, PathBuf};
use tracing::info;

/// Assets embarcados: modelo ONNX, brain_meta, script Python, OBJs premium.
#[derive(Embed)]
#[folder = "assets/models/onnx/"]
#[prefix = "onnx/"]
pub struct OnnxAssets;

#[derive(Embed)]
#[folder = "assets/models/premium/"]
#[prefix = "premium/"]
pub struct PremiumMeshes;

#[derive(Embed)]
#[folder = "scripts/ml/"]
#[prefix = "scripts/"]
pub struct ScriptAssets;

/// brain_meta.json embarcado diretamente.
pub const BRAIN_META_JSON: &[u8] = include_bytes!("../assets/models/brain_meta.json");

/// Paths dos assets extraidos em disco (necessarios pelo Python subprocess).
pub struct ExtractedAssets {
    pub model_path: PathBuf,
    pub meta_path: PathBuf,
    pub script_path: PathBuf,
    /// Diretorio base onde os assets foram extraidos.
    #[allow(dead_code)]
    pub base_dir: PathBuf,
}

/// Extrai assets para disco ao lado do executavel.
///
/// Cria `neuroscan_data/` ao lado do binario com:
///   - onnx/nnunet_brats_4ch.onnx
///   - brain_meta.json
///   - scripts/infer_tumor_3d.py
///
/// Idempotente: so extrai se o arquivo nao existe ou o tamanho mudou.
pub fn extract_assets() -> anyhow::Result<ExtractedAssets> {
    let base_dir = assets_base_dir();
    std::fs::create_dir_all(&base_dir)?;

    // ONNX model
    let onnx_dir = base_dir.join("onnx");
    std::fs::create_dir_all(&onnx_dir)?;
    let model_path = onnx_dir.join("nnunet_brats_4ch.onnx");
    extract_file::<OnnxAssets>("onnx/nnunet_brats_4ch.onnx", &model_path)?;

    // brain_meta.json
    let meta_path = base_dir.join("brain_meta.json");
    write_if_changed(&meta_path, BRAIN_META_JSON)?;

    // Python script
    let scripts_dir = base_dir.join("scripts");
    std::fs::create_dir_all(&scripts_dir)?;
    let script_path = scripts_dir.join("infer_tumor_3d.py");
    extract_file::<ScriptAssets>("scripts/infer_tumor_3d.py", &script_path)?;

    info!(base = %base_dir.display(), "assets extraidos para disco");

    Ok(ExtractedAssets {
        model_path,
        meta_path,
        script_path,
        base_dir,
    })
}

/// Retorna bytes de um OBJ premium embarcado (para carregar na GPU sem extrair).
pub fn premium_obj_bytes(filename: &str) -> Option<rust_embed::EmbeddedFile> {
    let key = format!("premium/{}", filename);
    PremiumMeshes::get(&key)
}

/// Diretorio base: ao lado do executavel, ou fallback para diretorio atual.
fn assets_base_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("neuroscan_data")))
        .unwrap_or_else(|| PathBuf::from("neuroscan_data"))
}

/// Extrai um arquivo de um Embed para disco se nao existe ou tamanho mudou.
fn extract_file<E: Embed>(embed_key: &str, dest: &Path) -> anyhow::Result<()> {
    let file = E::get(embed_key)
        .ok_or_else(|| anyhow::anyhow!("asset embarcado nao encontrado: {}", embed_key))?;

    write_if_changed(dest, &file.data)
}

/// Escreve bytes em disco somente se o arquivo nao existe ou o tamanho difere.
fn write_if_changed(dest: &Path, data: &[u8]) -> anyhow::Result<()> {
    if let Ok(meta) = std::fs::metadata(dest) {
        if meta.len() == data.len() as u64 {
            return Ok(());
        }
    }
    std::fs::write(dest, data)?;
    info!(path = %dest.display(), size = data.len(), "asset extraido");
    Ok(())
}
