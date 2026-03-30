//! Lancamento do pipeline de inferencia em thread de background.
//!
//! Usa o pipeline nativo Rust (src/pipeline/) com ort + nifti + marching cubes.
//! Sem dependencia de Python em runtime.

use std::sync::mpsc;

use tracing::{info, warn};

use super::{InferMsg, InferPhase};

/// Inicia o pipeline de inferencia nativo em thread de background.
///
/// Retorna um receiver de `InferMsg` para o render loop consumir.
/// O pipeline roda inteiramente em Rust via ort (ONNX Runtime nativo).
pub(crate) fn launch(input_path: &str, out_dir: &str) -> mpsc::Receiver<InferMsg> {
    let (tx, rx) = mpsc::channel::<InferMsg>();
    let input_path = input_path.to_string();
    let out_dir = out_dir.to_string();

    std::thread::spawn(move || {
        info!(input = %input_path, outdir = %out_dir, "iniciando pipeline nativo de inferencia");

        let model_path = "assets/models/onnx/nnunet_brats_4ch.onnx";
        let meta_path = "assets/models/brain_meta.json";

        // Callback de progresso: converte PipelineMsg -> InferMsg e envia pelo canal
        let tx_progress = tx.clone();
        let progress = Box::new(move |msg: crate::pipeline::PipelineMsg| {
            let infer_msg = match msg {
                crate::pipeline::PipelineMsg::Phase(text) => {
                    // Mapeia texto de fase para InferPhase
                    let phase = if text.contains("NIfTI")
                        || text.contains("Carregando")
                        || text.contains("ONNX")
                        || text.contains("modelo")
                    {
                        InferPhase::Preprocessing
                    } else if text.contains("Inferindo") || text.contains("segmentacao") {
                        InferPhase::Slicing
                    } else if text.contains("superficies") || text.contains("Extraindo") {
                        InferPhase::MarchingCubes
                    } else {
                        InferPhase::Preprocessing
                    };
                    InferMsg::Phase(phase)
                }
                crate::pipeline::PipelineMsg::Slice { current, total } => {
                    InferMsg::Slice { current, total }
                }
                crate::pipeline::PipelineMsg::Volume {
                    class_name,
                    volume_ml,
                } => {
                    let class = match class_name.as_str() {
                        "ET" => 1,
                        "SNFH" => 2,
                        "NETC" => 3,
                        _ => 0,
                    };
                    InferMsg::PartialVolume { class, volume_ml }
                }
            };
            let _ = tx_progress.send(infer_msg);
        });

        // Executa o pipeline nativo
        match crate::pipeline::run_pipeline(&input_path, model_path, meta_path, &out_dir, progress)
        {
            Ok(result) => {
                info!(
                    case_id = %result.case_id,
                    et_ml = result.et_volume_ml,
                    snfh_ml = result.snfh_volume_ml,
                    netc_ml = result.netc_volume_ml,
                    meshes = ?result.meshes_generated,
                    "pipeline nativo concluido com sucesso"
                );
                let _ = tx.send(InferMsg::Done(true));
            }
            Err(e) => {
                let err_msg = format!("{:#}", e);
                warn!(error = %err_msg, "pipeline nativo falhou");
                let _ = tx.send(InferMsg::Phase(InferPhase::Error(err_msg)));
                let _ = tx.send(InferMsg::Done(false));
            }
        }
    });

    rx
}

/// Parseia o payload de uma linha `NEUROSCAN:<payload>` e retorna o InferMsg correspondente.
///
/// Mantido para testes do protocolo e eventual fallback Python.
#[cfg(test)]
pub(super) fn parse_neuroscan_line(payload: &str) -> Option<InferMsg> {
    if payload == "DONE" {
        return Some(InferMsg::Phase(InferPhase::Done));
    }

    if let Some(rest) = payload.strip_prefix("PHASE:") {
        let phase = match rest {
            "preprocessing" => InferPhase::Preprocessing,
            "slicing" => InferPhase::Slicing,
            "marching_cubes" => InferPhase::MarchingCubes,
            "done" => InferPhase::Done,
            other => InferPhase::Error(format!("fase desconhecida: {other}")),
        };
        return Some(InferMsg::Phase(phase));
    }

    if let Some(rest) = payload.strip_prefix("SLICE:") {
        let parts: Vec<&str> = rest.splitn(2, ':').collect();
        if parts.len() == 2 {
            if let (Ok(current), Ok(total)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some(InferMsg::Slice { current, total });
            }
        }
        return None;
    }

    if let Some(rest) = payload.strip_prefix("VOLUME:") {
        let parts: Vec<&str> = rest.splitn(2, ':').collect();
        if parts.len() == 2 {
            let class: u8 = match parts[0] {
                "ET" => 1,
                "SNFH" => 2,
                "NETC" => 3,
                _ => return None,
            };
            if let Ok(volume_ml) = parts[1].parse::<f32>() {
                return Some(InferMsg::PartialVolume { class, volume_ml });
            }
        }
        return None;
    }

    if let Some(msg) = payload.strip_prefix("ERROR:") {
        return Some(InferMsg::Phase(InferPhase::Error(msg.to_string())));
    }

    None
}
