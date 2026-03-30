use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use tracing::{info, warn};

use super::{InferMsg, InferPhase};

/// Verifica se Python e as dependencias necessarias estao disponiveis.
///
/// Executa `python -c "import ..."` de forma sincrona (~100ms).
/// Retorna Ok com os providers ONNX disponiveis, ou Err com mensagem legivel.
pub(crate) fn check_python_env() -> Result<String, String> {
    let check_script = concat!(
        "import nibabel, onnxruntime, scipy, skimage; ",
        "import onnxruntime as ort; ",
        "print(','.join(ort.get_available_providers()))"
    );
    let output = Command::new("python")
        .args(["-c", check_script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let providers = String::from_utf8_lossy(&out.stdout).trim().to_string();
            info!(providers = %providers, "ambiente Python verificado");
            Ok(providers)
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(stderr = %stderr, "dependencias Python faltando");
            Err(
                "Dependencias Python faltando. Execute: pip install nibabel onnxruntime scipy scikit-image"
                    .to_string(),
            )
        }
        Err(e) => {
            warn!(error = %e, "Python nao encontrado");
            Err(format!(
                "Python nao encontrado ({}). Instale Python 3.10+ com as dependencias.",
                e
            ))
        }
    }
}

/// Inicia o subprocess Python de inferência e retorna um receiver de progresso.
///
/// Lê stdout linha a linha em thread dedicada, parseia o protocolo `NEUROSCAN:*`
/// e envia `InferMsg` para a thread principal sem bloquear o render loop.
pub(crate) fn launch(input_path: &str, out_dir: &str) -> mpsc::Receiver<InferMsg> {
    let (tx, rx) = mpsc::channel::<InferMsg>();
    let input_path = input_path.to_string();
    let out_dir = out_dir.to_string();

    std::thread::spawn(move || {
        info!(input = %input_path, outdir = %out_dir, "iniciando subprocess Python de inferencia");

        let child = Command::new("python")
            .args([
                "scripts/ml/infer_tumor_3d.py",
                "--input",
                &input_path,
                "--outdir",
                &out_dir,
                "--model",
                "assets/models/onnx/nnunet_brats_4ch.onnx",
                "--meta",
                "assets/models/brain_meta.json",
            ])
            // Garante flush imediato de stdout no Windows — defesa em profundidade
            // (o script Python ja usa flush=True em ns_print, mas o env var cobre
            // qualquer print() que escape sem flush explicito)
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "falha ao iniciar subprocess Python");
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        // Captura stdout e stderr do processo filho.
        // CRITICO: stderr DEVE ser drenado em thread separada para evitar deadlock.
        // Os warnings do ONNX Runtime CUDA (~KB de texto wide-char) enchem o pipe
        // buffer (64KB no Windows), bloqueando Python se o buffer nao for consumido.
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                warn!("stdout do subprocess Python nao disponivel");
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        // Drena stderr em thread separada para prevenir deadlock de pipe
        let stderr_handle = child.stderr.take().map(|mut stderr| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = std::io::Read::read_to_string(&mut stderr, &mut buf);
                buf
            })
        });

        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    warn!(error = %e, "erro ao ler stdout do subprocess");
                    break;
                }
            };

            // Processa linhas com o prefixo do protocolo
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

        // Coleta stderr da thread de drenagem
        let stderr_text = stderr_handle
            .and_then(|h| h.join().ok())
            .unwrap_or_default();

        // Espera o processo terminar e envia resultado final
        let success = match child.wait() {
            Ok(status) => {
                if !status.success() {
                    let relevant_lines: Vec<&str> = stderr_text
                        .lines()
                        .filter(|l| {
                            !l.is_empty()
                                && !l.contains("provider_bridge")
                                && !l.contains("pybind_state")
                                && !l.contains("CreateExecutionProviderFactory")
                                && !l.contains("onnxruntime_providers_cuda")
                        })
                        .collect();
                    let err_summary = relevant_lines
                        .last()
                        .copied()
                        .unwrap_or("erro desconhecido no subprocess Python")
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
                    "falha ao aguardar processo: {}",
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
        // DONE via protocolo (antes do exit code) — apenas sinaliza fase concluída
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
