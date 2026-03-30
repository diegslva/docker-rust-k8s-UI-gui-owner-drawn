//! Pipeline de inferencia nativo — substitui o subprocess Python.
//!
//! Componentes:
//! - nifti_loader: carrega volumes NIfTI (.nii.gz) 4 canais
//! - inference: sessao ONNX Runtime + inferencia slice-a-slice
//! - marching_cubes: extracao de superficies 3D
//! - mesh_export: escrita de OBJ + scan_meta.json

pub(crate) mod inference;
pub(crate) mod marching_cubes;
pub(crate) mod mesh_export;
pub(crate) mod nifti_loader;

/// Resultado do pipeline completo.
pub(crate) struct InferResult {
    pub case_id: String,
    pub et_volume_ml: f32,
    pub snfh_volume_ml: f32,
    pub netc_volume_ml: f32,
    pub total_volume_ml: f32,
    pub meshes_generated: Vec<String>,
    pub out_dir: String,
}

/// Mensagem de progresso emitida pelo pipeline.
pub(crate) enum PipelineMsg {
    Phase(String),
    Slice { current: u32, total: u32 },
    Volume { class_name: String, volume_ml: f32 },
}

/// Callback para progresso — projetado para ser usado com mpsc::Sender.
pub(crate) type ProgressFn = Box<dyn Fn(PipelineMsg) + Send>;

// Constantes do pipeline (replicadas do Python)
const INPUT_SIZE: usize = 256;
const MIN_VOXELS_PER_CLASS: usize = 200;
/// BraTS 1mm isotropico: 1 voxel = 1mm^3 = 0.001 mL
const VOXEL_TO_ML: f32 = 0.001;

/// Definicao das classes tumorais.
pub(crate) const TUMOR_CLASSES: &[(u8, &str, &str)] = &[
    (1, "ET", "tumor_et.obj"),
    (2, "SNFH", "tumor_snfh.obj"),
    (3, "NETC", "tumor_netc.obj"),
];

/// Executa o pipeline completo de forma sincrona.
///
/// Projetado para rodar em thread de background — nao bloqueia o render loop.
pub(crate) fn run_pipeline(
    input_path: &str,
    model_path: &str,
    meta_path: &str,
    out_dir: &str,
    progress: ProgressFn,
) -> anyhow::Result<InferResult> {
    use tracing::info;

    let case_id = std::path::Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .replace(".nii", "");

    // 1. Carregar volume NIfTI
    progress(PipelineMsg::Phase("Carregando volume NIfTI...".into()));
    let volume = nifti_loader::load_volume_4ch(input_path)?;
    let (h, w, d, _) = volume.dim();
    info!(h, w, d, "volume NIfTI carregado");

    // 2. Criar sessao ONNX
    progress(PipelineMsg::Phase("Inicializando modelo ONNX...".into()));
    let mut session = inference::create_session(model_path)?;

    // 3. Inferencia slice-a-slice
    progress(PipelineMsg::Phase("Inferindo segmentacao...".into()));
    let mask = inference::infer_volume(&mut session, &volume, &|cur, tot| {
        progress(PipelineMsg::Slice {
            current: cur,
            total: tot,
        });
    })?;

    // 4. Carregar brain_meta.json
    let (center, scale, upsample) = mesh_export::load_brain_meta(meta_path)?;

    // 5. Extrair meshes por classe tumoral
    progress(PipelineMsg::Phase("Extraindo superficies 3D...".into()));
    std::fs::create_dir_all(out_dir)?;

    let mut et_ml = 0.0_f32;
    let mut snfh_ml = 0.0_f32;
    let mut netc_ml = 0.0_f32;
    let mut generated = Vec::new();

    for &(class_id, name, obj_name) in TUMOR_CLASSES {
        let voxel_count = mask.iter().filter(|&&v| v == class_id).count();
        let volume_ml = (voxel_count as f32 * VOXEL_TO_ML * 100.0).round() / 100.0;

        match class_id {
            1 => et_ml = volume_ml,
            2 => snfh_ml = volume_ml,
            3 => netc_ml = volume_ml,
            _ => {}
        }
        progress(PipelineMsg::Volume {
            class_name: name.to_string(),
            volume_ml,
        });

        if voxel_count < MIN_VOXELS_PER_CLASS {
            info!(
                class = name,
                voxels = voxel_count,
                "classe com poucos voxels — mesh nao gerado"
            );
            continue;
        }

        if let Some((verts, faces)) =
            marching_cubes::extract_mesh(&mask, class_id, center, scale, upsample)
        {
            let normals = mesh_export::compute_vertex_normals(&verts, &faces);
            let obj_path = format!("{}/{}", out_dir, obj_name);
            mesh_export::write_obj(
                &obj_path,
                &verts,
                &normals,
                &faces,
                &format!("Tumor {} -- NeuroScan AI", name),
            )?;
            generated.push(name.to_string());
            info!(
                class = name,
                vertices = verts.len(),
                faces = faces.len(),
                "mesh gerado"
            );
        }
    }

    let total_ml = (et_ml + snfh_ml + netc_ml * 100.0).round() / 100.0;

    // 6. Escrever scan_meta.json
    mesh_export::write_scan_meta(out_dir, &case_id, et_ml, snfh_ml, netc_ml)?;

    info!(
        case_id,
        et_ml,
        snfh_ml,
        netc_ml,
        total_ml,
        generated = ?generated,
        "pipeline nativo concluido"
    );

    Ok(InferResult {
        case_id,
        et_volume_ml: et_ml,
        snfh_volume_ml: snfh_ml,
        netc_volume_ml: netc_ml,
        total_volume_ml: total_ml,
        meshes_generated: generated,
        out_dir: out_dir.to_string(),
    })
}
