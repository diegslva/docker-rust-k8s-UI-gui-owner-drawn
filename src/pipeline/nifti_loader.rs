//! Carregamento de volumes NIfTI (.nii.gz) 4 canais MRI.

use anyhow::{Context, bail};
use ndarray::{Array3, Array4, Ix4};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use tracing::info;

/// Carrega um volume NIfTI 4-canais (FLAIR, T1w, T1ce, T2w).
///
/// Retorna Array4<f32> com shape (H, W, D, 4).
pub(crate) fn load_volume_4ch(path: &str) -> anyhow::Result<Array4<f32>> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("falha ao abrir NIfTI: {}", path))?;

    let header = obj.header();
    let dims = header.dim;
    info!(
        dims = ?&dims[..],
        "NIfTI header"
    );

    let volume = obj
        .into_volume()
        .into_ndarray::<f32>()
        .context("falha ao converter NIfTI para ndarray f32")?;

    let shape = volume.shape().to_vec();
    info!(shape = ?shape, "NIfTI shape");

    // BraTS formato: (H, W, D, 4) — 4 canais MRI
    let arr4 = volume
        .into_dimensionality::<Ix4>()
        .map_err(|e| anyhow::anyhow!("NIfTI nao e 4D: {}. Shape esperado: (H, W, D, 4)", e))?;

    let (_, _, _, channels) = arr4.dim();
    if channels < 4 {
        bail!(
            "NIfTI tem {} canais, esperado 4 (FLAIR, T1w, T1ce, T2w)",
            channels
        );
    }

    Ok(arr4)
}

/// Z-score normaliza um volume 3D (apenas voxels nao-zero).
///
/// Mascara cerebral: voxels > 0 sao normalizados; voxels == 0 permanecem zero.
pub(crate) fn zscore_normalize(vol: &Array3<f32>) -> Array3<f32> {
    let mut out = Array3::<f32>::zeros(vol.raw_dim());

    // Coleta estatisticas dos voxels nao-zero
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0_u64;

    for &v in vol.iter() {
        if v > 0.0 {
            let vd = v as f64;
            sum += vd;
            sum_sq += vd * vd;
            count += 1;
        }
    }

    if count == 0 {
        return out;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - mean * mean;
    let std_dev = variance.sqrt().max(1e-8);

    // Aplica normalizacao
    for (o, &v) in out.iter_mut().zip(vol.iter()) {
        if v > 0.0 {
            *o = ((v as f64 - mean) / std_dev) as f32;
        }
    }

    out
}
