//! Inferencia ONNX Runtime via ort crate — substitui o subprocess Python.

use anyhow::Context;
use ndarray::{Array2, Array3, Array4, s};
use tracing::{debug, info};

use super::nifti_loader::zscore_normalize;

const INPUT_SIZE: usize = 256;

/// Cria uma sessao ONNX Runtime a partir de um arquivo .onnx.
///
/// Seleciona o melhor execution provider por plataforma:
/// - Windows: DirectML (GPU via DirectX 12, sem CUDA Toolkit)
/// - Linux/Jetson: TensorRT > CUDA > CPU fallback
pub(crate) fn create_session(model_path: &str) -> anyhow::Result<ort::session::Session> {
    let builder =
        ort::session::Session::builder().context("falha ao criar builder de sessao ONNX")?;

    let mut builder = register_gpu_providers(builder);

    let session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("falha ao carregar modelo ONNX: {}", model_path))?;

    info!("sessao ONNX criada a partir de {}", model_path);

    Ok(session)
}

/// Registra execution providers GPU por plataforma.
fn register_gpu_providers(
    builder: ort::session::builder::SessionBuilder,
) -> ort::session::builder::SessionBuilder {
    use tracing::warn;

    #[cfg(target_os = "windows")]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        eprintln!("[NeuroScan] Registrando DirectML execution provider (Windows GPU)...");
        info!("registrando DirectML execution provider (Windows GPU)");
        match builder.with_execution_providers([DirectMLExecutionProvider::default().build()]) {
            Ok(b) => {
                eprintln!("[NeuroScan] DirectML ATIVO — inferencia via GPU");
                return b;
            }
            Err(e) => {
                eprintln!("[NeuroScan] DirectML FALHOU: {} — usando CPU", e.message());
                warn!("DirectML nao disponivel, usando CPU: {}", e.message());
                return e.recover();
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
        info!("registrando TensorRT/CUDA execution providers (Linux GPU)");
        match builder.with_execution_providers([
            TensorRTExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
        ]) {
            Ok(b) => return b,
            Err(e) => {
                warn!(
                    "GPU execution providers nao disponiveis, usando CPU: {}",
                    e.message()
                );
                return e.recover();
            }
        }
    }
}

/// Executa inferencia slice-a-slice em um volume 4-canais.
///
/// Retorna mascara 3D (H, W, D) com labels de classe (0=background, 1=ET, 2=SNFH, 3=NETC).
pub(crate) fn infer_volume(
    session: &mut ort::session::Session,
    volume: &Array4<f32>,
    progress: &dyn Fn(u32, u32),
) -> anyhow::Result<Array3<u8>> {
    let (h, w, d, _) = volume.dim();
    let mut mask = Array3::<u8>::zeros((h, w, d));

    // Z-score normaliza cada canal
    let mut norm_channels = Vec::with_capacity(4);
    for c in 0..4 {
        let channel = volume.slice(s![.., .., .., c]).to_owned();
        norm_channels.push(zscore_normalize(&channel));
    }

    info!(slices = d, "iniciando inferencia slice-a-slice");
    progress(0, d as u32);

    for z in 0..d {
        // Extrai 4 canais da fatia z
        let slices: Vec<Array2<f32>> = (0..4)
            .map(|c| norm_channels[c].slice(s![.., .., z]).to_owned())
            .collect();

        // Verifica se a fatia e vazia (todos os canais ~zero)
        let is_empty = slices
            .iter()
            .all(|s| s.iter().copied().fold(0.0_f32, f32::max) < 1e-6);
        if is_empty {
            progress(z as u32 + 1, d as u32);
            continue;
        }

        // Redimensiona cada canal para INPUT_SIZE x INPUT_SIZE
        let resized: Vec<Array2<f32>> = slices
            .iter()
            .map(|s| bilinear_resize(s, INPUT_SIZE, INPUT_SIZE))
            .collect();

        // Empilha em tensor (1, 4, H, W)
        let mut tensor = ndarray::Array4::<f32>::zeros((1, 4, INPUT_SIZE, INPUT_SIZE));
        for (c, ch) in resized.iter().enumerate() {
            for (i, row) in ch.outer_iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    tensor[[0, c, i, j]] = val;
                }
            }
        }

        // Cria tensor ONNX e executa inferencia
        let flat: Vec<f32> = tensor.iter().copied().collect();
        let input_tensor = ort::value::Tensor::<f32>::from_array((
            vec![1_i64, 4, INPUT_SIZE as i64, INPUT_SIZE as i64],
            flat,
        ))
        .with_context(|| format!("falha ao criar tensor para fatia {}", z))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .with_context(|| format!("falha na inferencia da fatia {}", z))?;

        // Output shape: (1, num_classes, H, W) — argmax sobre dim 1
        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("falha ao extrair tensor de saida")?;
        let out_dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        debug!(z, shape = ?out_dims, "output da fatia");

        let num_classes = out_dims[1];
        let out_h = out_dims[2];
        let out_w = out_dims[3];

        // Argmax por pixel (acesso flat: [batch, class, row, col])
        let mut pred_small = Array2::<u8>::zeros((out_h, out_w));
        for i in 0..out_h {
            for j in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_cls = 0_u8;
                for c in 0..num_classes {
                    let idx = c * out_h * out_w + i * out_w + j;
                    let val = out_data[idx];
                    if val > max_val {
                        max_val = val;
                        max_cls = c as u8;
                    }
                }
                pred_small[[i, j]] = max_cls;
            }
        }

        // Redimensiona predicao de volta ao tamanho original (nearest-neighbor)
        let pred_full = if out_h != h || out_w != w {
            nearest_resize_u8(&pred_small, h, w)
        } else {
            pred_small
        };

        // Armazena na mascara 3D
        for i in 0..h {
            for j in 0..w {
                mask[[i, j, z]] = pred_full[[i, j]];
            }
        }

        progress(z as u32 + 1, d as u32);
    }

    // Distribuicao de classes
    let mut counts = [0_u64; 4];
    for &v in mask.iter() {
        if (v as usize) < counts.len() {
            counts[v as usize] += 1;
        }
    }
    info!(
        bg = counts[0],
        et = counts[1],
        snfh = counts[2],
        netc = counts[3],
        "distribuicao de classes na mascara"
    );

    Ok(mask)
}

/// Redimensionamento bilinear 2D de uma Array2<f32>.
fn bilinear_resize(src: &Array2<f32>, new_h: usize, new_w: usize) -> Array2<f32> {
    let (src_h, src_w) = src.dim();
    if src_h == new_h && src_w == new_w {
        return src.clone();
    }

    let mut dst = Array2::<f32>::zeros((new_h, new_w));
    let scale_y = src_h as f32 / new_h as f32;
    let scale_x = src_w as f32 / new_w as f32;

    for i in 0..new_h {
        for j in 0..new_w {
            let sy = i as f32 * scale_y;
            let sx = j as f32 * scale_x;

            let y0 = (sy.floor() as usize).min(src_h - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let x0 = (sx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);

            let fy = sy - y0 as f32;
            let fx = sx - x0 as f32;

            let val = src[[y0, x0]] * (1.0 - fy) * (1.0 - fx)
                + src[[y1, x0]] * fy * (1.0 - fx)
                + src[[y0, x1]] * (1.0 - fy) * fx
                + src[[y1, x1]] * fy * fx;

            dst[[i, j]] = val;
        }
    }

    dst
}

/// Redimensionamento nearest-neighbor para mascaras (u8).
fn nearest_resize_u8(src: &Array2<u8>, new_h: usize, new_w: usize) -> Array2<u8> {
    let (src_h, src_w) = src.dim();
    let mut dst = Array2::<u8>::zeros((new_h, new_w));
    let scale_y = src_h as f32 / new_h as f32;
    let scale_x = src_w as f32 / new_w as f32;

    for i in 0..new_h {
        for j in 0..new_w {
            let sy = ((i as f32 * scale_y) as usize).min(src_h - 1);
            let sx = ((j as f32 * scale_x) as usize).min(src_w - 1);
            dst[[i, j]] = src[[sy, sx]];
        }
    }

    dst
}
