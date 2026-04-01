//! Loader de volumes MRI (.npy) para o slice viewer 3D.
//!
//! O Python (infer_tumor_3d.py) salva o canal FLAIR como uint8 em formato numpy
//! com shape (D, H, W) — layout compativel com wgpu write_texture.

use anyhow::{Context, bail};
use glam::Vec3;
use tracing::info;

/// Volume MRI carregado em memoria, pronto para upload na GPU como textura 3D.
pub struct VolumeData {
    /// Voxels uint8, layout (D, H, W) row-major.
    pub data: Vec<u8>,
    /// Dimensoes do volume: [width, height, depth].
    pub dims: [u32; 3],
    /// Centro do cerebro no espaco de voxel (de brain_meta.json).
    pub center: [f64; 3],
    /// Fator de escala para normalizacao dos meshes.
    pub scale: f64,
    /// Fator de upsample aplicado pelo Marching Cubes.
    pub upsample_factor: f64,
}

impl VolumeData {
    /// Carrega um volume .npy (uint8, shape D,H,W) e metadados de brain_meta.json.
    pub fn load(npy_path: &str, meta_path: &str) -> anyhow::Result<Self> {
        let npy_bytes =
            std::fs::read(npy_path).with_context(|| format!("falha ao ler {npy_path}"))?;

        let (dims, data) = parse_npy_u8(&npy_bytes)?;

        // dims do .npy: [D, H, W] (transposto no Python)
        let depth = dims[0];
        let height = dims[1];
        let width = dims[2];

        // brain_meta.json
        let meta_text = std::fs::read_to_string(meta_path)
            .with_context(|| format!("falha ao ler {meta_path}"))?;
        let meta: serde_json::Value =
            serde_json::from_str(&meta_text).context("falha ao parsear brain_meta.json")?;

        let center = [
            meta["center"][0].as_f64().unwrap_or(0.0),
            meta["center"][1].as_f64().unwrap_or(0.0),
            meta["center"][2].as_f64().unwrap_or(0.0),
        ];
        let scale = meta["scale"].as_f64().unwrap_or(1.0);
        let upsample_factor = meta["upsample_factor"].as_f64().unwrap_or(1.0);

        info!(
            width,
            height,
            depth,
            size_mb = data.len() as f64 / 1e6,
            "volume MRI carregado"
        );

        Ok(Self {
            data,
            dims: [width, height, depth],
            center,
            scale,
            upsample_factor,
        })
    }

    /// Retorna os bounds do volume no espaco normalizado dos meshes.
    ///
    /// Os meshes sao normalizados como: `(voxel * upsample - center) / scale`.
    /// O volume original tem dims (H, W, D) no espaco de voxel.
    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        let [w, h, d] = self.dims;
        let up = self.upsample_factor;
        let c = Vec3::new(
            self.center[0] as f32,
            self.center[1] as f32,
            self.center[2] as f32,
        );
        let s = self.scale as f32;

        let world_min = (Vec3::ZERO - c) / s;
        let world_max = (Vec3::new(
            w as f32 * up as f32,
            h as f32 * up as f32,
            d as f32 * up as f32,
        ) - c)
            / s;

        (world_min, world_max)
    }
}

/// Plano anatomico de corte.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SlicePlane {
    /// XY plano, varia Z (mais comum para cerebro).
    Axial,
    /// XZ plano, varia Y (vista frontal).
    Coronal,
    /// YZ plano, varia X (vista lateral).
    Sagittal,
}

/// Parser minimo de .npy (numpy format 1.0, uint8, C-contiguous).
///
/// Formato: magic (6 bytes) + version (2 bytes) + header_len (2 bytes LE)
/// + header (ASCII dict) + raw data.
fn parse_npy_u8(bytes: &[u8]) -> anyhow::Result<(Vec<u32>, Vec<u8>)> {
    if bytes.len() < 10 {
        bail!("arquivo .npy muito pequeno");
    }

    // Magic: \x93NUMPY
    if &bytes[0..6] != b"\x93NUMPY" {
        bail!("magic .npy invalido");
    }

    let major = bytes[6];
    let _minor = bytes[7];

    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        // Version 2+: 4-byte header length
        if bytes.len() < 12 {
            bail!(".npy v2 header truncado");
        }
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };

    let header_start = if major == 1 { 10 } else { 12 };
    let data_start = header_start + header_len;

    if bytes.len() < data_start {
        bail!("header .npy truncado");
    }

    let header =
        std::str::from_utf8(&bytes[header_start..data_start]).context("header .npy nao e UTF-8")?;

    // Extrair shape do header: 'shape': (D, H, W)
    let dims = parse_shape(header)?;

    // Verificar dtype uint8
    if !header.contains("|u1") && !header.contains("uint8") {
        bail!("dtype .npy nao e uint8: {header}");
    }

    let expected_size: usize = dims.iter().map(|&d| d as usize).product();
    let data = &bytes[data_start..];

    if data.len() < expected_size {
        bail!(
            "dados .npy truncados: esperado {expected_size}, encontrado {}",
            data.len()
        );
    }

    Ok((dims, data[..expected_size].to_vec()))
}

/// Extrai shape do header .npy: "'shape': (155, 240, 240)" -> [155, 240, 240].
fn parse_shape(header: &str) -> anyhow::Result<Vec<u32>> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .context("campo 'shape' nao encontrado no header .npy")?;

    let paren_start = header[shape_start..]
        .find('(')
        .context("parenteses de shape nao encontrados")?
        + shape_start;
    let paren_end = header[paren_start..]
        .find(')')
        .context("parenteses de shape nao fechados")?
        + paren_start;

    let shape_str = &header[paren_start + 1..paren_end];
    let dims: Vec<u32> = shape_str
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    if dims.len() != 3 {
        bail!("shape .npy nao e 3D: {:?}", dims);
    }

    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shape_standard() {
        let header = "{'descr': '|u1', 'fortran_order': False, 'shape': (155, 240, 240), }";
        let dims = parse_shape(header).unwrap();
        assert_eq!(dims, vec![155, 240, 240]);
    }

    #[test]
    fn parse_shape_trailing_comma() {
        let header = "{'shape': (155, 240, 240,), 'descr': '|u1'}";
        let dims = parse_shape(header).unwrap();
        assert_eq!(dims, vec![155, 240, 240]);
    }
}
