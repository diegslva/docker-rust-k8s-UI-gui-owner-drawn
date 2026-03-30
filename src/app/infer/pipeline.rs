use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use tracing::{info, warn};

use super::{InferMsg, InferPhase};

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
            .stderr(Stdio::null())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "falha ao iniciar subprocess Python");
                let _ = tx.send(InferMsg::Done(false));
                return;
            }
        };

        // Captura stdout do processo filho e parseia linha a linha
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

            // Só processa linhas com o prefixo do protocolo
            if let Some(payload) = line.strip_prefix("NEUROSCAN:") {
                if let Some(msg) = parse_neuroscan_line(payload) {
                    // Se o canal foi fechado (receiver dropado), para de enviar
                    if tx.send(msg).is_err() {
                        break;
                    }
                } else {
                    // Linha com prefixo mas payload desconhecido — log e continua
                    warn!(line = %line, "linha NEUROSCAN desconhecida");
                }
            }
            // Linhas sem o prefixo são descartadas silenciosamente
        }

        // Espera o processo terminar e envia resultado final
        let success = match child.wait() {
            Ok(status) => {
                if !status.success() {
                    warn!(code = ?status.code(), "subprocess Python terminou com erro");
                }
                status.success()
            }
            Err(e) => {
                warn!(error = %e, "erro ao aguardar subprocess Python");
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
