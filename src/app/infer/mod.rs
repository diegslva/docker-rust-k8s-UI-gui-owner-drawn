pub(crate) mod pipeline;

/// Mensagem enviada da thread de inferência para a thread principal.
#[derive(Debug)]
pub(crate) enum InferMsg {
    /// Progresso por fatia axial (1-indexado).
    Slice { current: u32, total: u32 },
    /// Volume parcial de uma classe assim que calculado.
    /// class: 1=ET  2=SNFH  3=NETC
    PartialVolume { class: u8, volume_ml: f32 },
    /// Mudança de fase do pipeline.
    Phase(InferPhase),
    /// Pipeline concluído (true=sucesso, false=erro).
    Done(bool),
}

/// Fases nominais do pipeline de inferência.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum InferPhase {
    Preprocessing,
    Slicing,
    MarchingCubes,
    Done,
    Error(String),
}

/// Estado de progresso acumulado — atualizado pelo loop de render a partir dos InferMsg.
#[derive(Clone)]
pub(crate) struct InferProgress {
    pub(crate) phase: InferPhase,
    pub(crate) current_slice: u32,
    pub(crate) total_slices: u32,
    pub(crate) et_volume_ml: f32,
    pub(crate) snfh_volume_ml: f32,
    pub(crate) netc_volume_ml: f32,
    /// Segundos decorridos desde o início.
    pub(crate) elapsed_secs: f32,
    /// Tempo acumulado para animações da tela de inferência.
    pub(crate) anim_t: f32,
}

impl Default for InferProgress {
    fn default() -> Self {
        Self {
            phase: InferPhase::Preprocessing,
            current_slice: 0,
            total_slices: 155,
            et_volume_ml: 0.0,
            snfh_volume_ml: 0.0,
            netc_volume_ml: 0.0,
            elapsed_secs: 0.0,
            anim_t: 0.0,
        }
    }
}
