//! Exportacao de meshes OBJ e metadados scan_meta.json.

use anyhow::Context;
use std::io::Write;
use tracing::info;

/// Carrega brain_meta.json (center, scale, upsample_factor).
pub(crate) fn load_brain_meta(path: &str) -> anyhow::Result<([f64; 3], f64, f64)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("falha ao ler brain_meta.json: {}", path))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).context("falha ao parsear brain_meta.json")?;

    let center_arr = v["center"]
        .as_array()
        .context("campo 'center' ausente ou invalido")?;
    let center = [
        center_arr[0].as_f64().unwrap_or(0.0),
        center_arr[1].as_f64().unwrap_or(0.0),
        center_arr[2].as_f64().unwrap_or(0.0),
    ];
    let scale = v["scale"].as_f64().unwrap_or(1.0);
    let upsample = v
        .get("upsample_factor")
        .and_then(|u| u.as_f64())
        .unwrap_or(1.0);

    info!(
        center = ?center,
        scale,
        upsample,
        "brain_meta carregado"
    );

    Ok((center, scale, upsample))
}

/// Calcula normais por vertice (media ponderada por area das faces adjacentes).
pub(crate) fn compute_vertex_normals(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0_f32; 3]; vertices.len()];

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Normal da face (cross product)
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let fn_ = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Acumula nos vertices da face
        for &idx in &[i0, i1, i2] {
            normals[idx][0] += fn_[0];
            normals[idx][1] += fn_[1];
            normals[idx][2] += fn_[2];
        }
    }

    // Normaliza
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-10 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }

    normals
}

/// Escreve um arquivo OBJ com vertices, normais e faces.
pub(crate) fn write_obj(
    path: &str,
    vertices: &[[f32; 3]],
    normals: &[[f32; 3]],
    faces: &[[u32; 3]],
    comment: &str,
) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("falha ao criar OBJ: {}", path))?,
    );

    if !comment.is_empty() {
        writeln!(f, "# {}", comment)?;
    }
    writeln!(f, "# {} vertices, {} faces\n", vertices.len(), faces.len())?;

    for v in vertices {
        writeln!(f, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2])?;
    }
    writeln!(f)?;

    for n in normals {
        writeln!(f, "vn {:.6} {:.6} {:.6}", n[0], n[1], n[2])?;
    }
    writeln!(f)?;

    for tri in faces {
        let (i, j, k) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        writeln!(f, "f {}//{} {}//{} {}//{}", i, i, j, j, k, k)?;
    }

    f.flush()?;
    info!(
        path,
        vertices = vertices.len(),
        faces = faces.len(),
        "OBJ escrito"
    );
    Ok(())
}

/// Escreve scan_meta.json com informacoes de volume.
pub(crate) fn write_scan_meta(
    out_dir: &str,
    case_id: &str,
    et_ml: f32,
    snfh_ml: f32,
    netc_ml: f32,
) -> anyhow::Result<()> {
    let total = ((et_ml + snfh_ml + netc_ml) * 100.0).round() / 100.0;

    let meta = serde_json::json!({
        "case_id": case_id,
        "dataset": "Paciente -- Inferencia NeuroScan AI",
        "modalities": "FLAIR . T1w . T1ce . T2w",
        "et_volume_ml": et_ml,
        "snfh_volume_ml": snfh_ml,
        "netc_volume_ml": netc_ml,
        "total_volume_ml": total,
    });

    let path = format!("{}/scan_meta.json", out_dir);
    let text = serde_json::to_string_pretty(&meta)?;
    std::fs::write(&path, &text)
        .with_context(|| format!("falha ao escrever scan_meta.json: {}", path))?;

    info!(path, "scan_meta.json escrito");
    Ok(())
}
