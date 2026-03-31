//! Lancamento do pipeline de inferencia via Python subprocess.
//!
//! Usa o script `infer_tumor_3d.py` com onnxruntime-gpu para inferencia ONNX.
//! O script e extraido do binario pelo modulo `embedded` e executado via subprocess.
//! GPU automatica: TensorRT > CUDA > CPU (Python ONNX Runtime).

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use tracing::{info, warn};

use super::{InferMsg, InferPhase};

/// Inicia o subprocess Python de inferencia e retorna um receiver de progresso.
///
/// Le stdout linha a linha em thread dedicada, parseia o protocolo `NEUROSCAN:*`
/// e envia `InferMsg` para a thread principal sem bloquear o render loop.
/// Stderr e drenado em thread separada para evitar deadlock de pipe no Windows.
pub(crate) fn launch(input_path: &str, out_dir: &str) -> mpsc::Receiver<InferMsg> {
    let (tx, rx) = mpsc::channel::<InferMsg>();
    let input_path = input_path.to_string();
    let out_dir = out_dir.to_string();

    std::thread::spawn(move || {
        info!(input = %input_path, outdir = %out_dir, "iniciando subprocess Python de inferencia");

        // Extrair assets embarcados (ONNX model, brain_meta, script Python)
        let assets = match crate::embedded::extract_assets() {
            Ok(a) => a,
            Err(e) => {
                warn!(error = %e, "falha ao extrair assets embarcados");
                let _ = tx.send(InferMsg::Phase(InferPhase::Error(format!(
                    "Falha ao extrair assets: {}",
                    e
                ))));
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        // Garantir ambiente Python (detecta sistema, cria venv, ou baixa standalone)
        let _ = tx.send(InferMsg::Phase(InferPhase::PythonSetup));
        let tx_setup = tx.clone();
        let python_env = match crate::python_env::ensure_python_env(Some(Box::new(
            move |msg: &str| {
                let _ = tx_setup.send(InferMsg::SetupStatus(msg.to_string()));
            },
        ))) {
            Ok(env) => {
                info!(source = %env.source, python = %env.python_bin.display(), "ambiente Python pronto");
                env
            }
            Err(e) => {
                warn!(error = %e, "falha ao configurar ambiente Python");
                let _ = tx.send(InferMsg::Phase(InferPhase::Error(e)));
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };
        let python = python_env.python_bin.to_string_lossy().to_string();

        let script_path = assets.script_path.to_string_lossy().to_string();
        let model_path = assets.model_path.to_string_lossy().to_string();
        let meta_path = assets.meta_path.to_string_lossy().to_string();

        let child = Command::new(&python)
            .args([
                &script_path,
                "--input",
                &input_path,
                "--outdir",
                &out_dir,
                "--model",
                &model_path,
                "--meta",
                &meta_path,
            ])
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "falha ao iniciar subprocess Python");
                let _ = tx.send(InferMsg::Phase(InferPhase::Error(format!(
                    "Falha ao iniciar Python: {}",
                    e
                ))));
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        // Drenar stderr em thread separada — evita deadlock de pipe no Windows.
        // ONNX Runtime escreve KB de warnings em stderr (CUDA, TensorRT probing).
        let stderr = child.stderr.take();
        let stderr_handle = std::thread::spawn(move || {
            if let Some(mut stderr) = stderr {
                let mut buf = Vec::new();
                let _ = stderr.read_to_end(&mut buf);
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            }
        });

        // Captura stdout e parseia protocolo NEUROSCAN
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                warn!("stdout do subprocess Python nao disponivel");
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    warn!(error = %e, "erro ao ler stdout do subprocess");
                    break;
                }
            };

            if let Some(payload) = line.strip_prefix("NEUROSCAN:") {
                if let Some(msg) = parse_neuroscan_line(payload) {
                    if tx.send(msg).is_err() {
                        break;
                    }
                } else {
                    warn!(line = %line, "linha NEUROSCAN desconhecida");
                }
            }
        }

        // Espera o processo terminar
        let success = match child.wait() {
            Ok(status) => {
                if !status.success() {
                    let stderr_text = stderr_handle.join().unwrap_or_default();
                    let relevant_lines: Vec<&str> = stderr_text
                        .lines()
                        .filter(|l| {
                            !l.is_empty()
                                && !l.contains("provider_bridge")
                                && !l.contains("pybind_state")
                        })
                        .collect();
                    let err_summary = relevant_lines
                        .last()
                        .copied()
                        .unwrap_or("erro desconhecido")
                        .to_string();
                    warn!(
                        code = ?status.code(),
                        stderr = %err_summary,
                        "subprocess Python terminou com erro"
                    );
                    let _ = tx.send(InferMsg::Phase(InferPhase::Error(err_summary)));
                }
                status.success()
            }
            Err(e) => {
                warn!(error = %e, "erro ao aguardar subprocess Python");
                let _ = tx.send(InferMsg::Phase(InferPhase::Error(format!(
                    "Falha ao aguardar processo: {}",
                    e
                ))));
                false
            }
        };

        let _ = tx.send(InferMsg::Done(success));
        info!(success, "subprocess Python de inferencia finalizado");
    });

    rx
}

/// Parseia o payload de uma linha `NEUROSCAN:<payload>` e retorna o InferMsg correspondente.
///
/// Formatos esperados:
///   PHASE:preprocessing | PHASE:slicing | PHASE:marching_cubes | PHASE:done
///   SLICE:<current>:<total>
///   VOLUME:ET:<ml> | VOLUME:SNFH:<ml> | VOLUME:NETC:<ml>
///   DONE
///   ERROR:<mensagem>
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
