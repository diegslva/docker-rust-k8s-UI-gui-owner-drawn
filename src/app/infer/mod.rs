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

impl InferProgress {
    /// Aplica uma mensagem ao estado de progresso acumulado.
    pub(crate) fn apply(&mut self, msg: &InferMsg) {
        match msg {
            InferMsg::Phase(p) => self.phase = p.clone(),
            InferMsg::Slice { current, total } => {
                self.current_slice = *current;
                if *total > 0 {
                    self.total_slices = *total;
                }
            }
            InferMsg::PartialVolume { class, volume_ml } => match class {
                1 => self.et_volume_ml = *volume_ml,
                2 => self.snfh_volume_ml = *volume_ml,
                3 => self.netc_volume_ml = *volume_ml,
                _ => {}
            },
            InferMsg::Done(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pipeline::parse_neuroscan_line;
    use super::*;

    // ── parse_neuroscan_line ─────────────────────────────────────────

    #[test]
    fn parse_phase_preprocessing() {
        let msg = parse_neuroscan_line("PHASE:preprocessing").unwrap();
        assert!(matches!(msg, InferMsg::Phase(InferPhase::Preprocessing)));
    }

    #[test]
    fn parse_phase_slicing() {
        let msg = parse_neuroscan_line("PHASE:slicing").unwrap();
        assert!(matches!(msg, InferMsg::Phase(InferPhase::Slicing)));
    }

    #[test]
    fn parse_phase_marching_cubes() {
        let msg = parse_neuroscan_line("PHASE:marching_cubes").unwrap();
        assert!(matches!(msg, InferMsg::Phase(InferPhase::MarchingCubes)));
    }

    #[test]
    fn parse_phase_done_keyword() {
        let msg = parse_neuroscan_line("DONE").unwrap();
        assert!(matches!(msg, InferMsg::Phase(InferPhase::Done)));
    }

    #[test]
    fn parse_slice_progress() {
        let msg = parse_neuroscan_line("SLICE:78:155").unwrap();
        match msg {
            InferMsg::Slice { current, total } => {
                assert_eq!(current, 78);
                assert_eq!(total, 155);
            }
            other => panic!("esperado Slice, recebido {:?}", other),
        }
    }

    #[test]
    fn parse_slice_first() {
        let msg = parse_neuroscan_line("SLICE:0:155").unwrap();
        match msg {
            InferMsg::Slice { current, total } => {
                assert_eq!(current, 0);
                assert_eq!(total, 155);
            }
            other => panic!("esperado Slice, recebido {:?}", other),
        }
    }

    #[test]
    fn parse_volume_et() {
        let msg = parse_neuroscan_line("VOLUME:ET:2.34").unwrap();
        match msg {
            InferMsg::PartialVolume { class, volume_ml } => {
                assert_eq!(class, 1);
                assert!((volume_ml - 2.34).abs() < 0.01);
            }
            other => panic!("esperado PartialVolume, recebido {:?}", other),
        }
    }

    #[test]
    fn parse_volume_snfh() {
        let msg = parse_neuroscan_line("VOLUME:SNFH:8.76").unwrap();
        match msg {
            InferMsg::PartialVolume { class, volume_ml } => {
                assert_eq!(class, 2);
                assert!((volume_ml - 8.76).abs() < 0.01);
            }
            other => panic!("esperado PartialVolume, recebido {:?}", other),
        }
    }

    #[test]
    fn parse_volume_netc() {
        let msg = parse_neuroscan_line("VOLUME:NETC:3.21").unwrap();
        match msg {
            InferMsg::PartialVolume { class, volume_ml } => {
                assert_eq!(class, 3);
                assert!((volume_ml - 3.21).abs() < 0.01);
            }
            other => panic!("esperado PartialVolume, recebido {:?}", other),
        }
    }

    #[test]
    fn parse_error_message() {
        let msg = parse_neuroscan_line("ERROR:modelo nao encontrado").unwrap();
        assert!(matches!(
            msg,
            InferMsg::Phase(InferPhase::Error(ref e)) if e == "modelo nao encontrado"
        ));
    }

    #[test]
    fn parse_unknown_payload_returns_none() {
        assert!(parse_neuroscan_line("GARBAGE:data").is_none());
    }

    #[test]
    fn parse_slice_invalid_numbers_returns_none() {
        assert!(parse_neuroscan_line("SLICE:abc:def").is_none());
    }

    #[test]
    fn parse_volume_unknown_class_returns_none() {
        assert!(parse_neuroscan_line("VOLUME:XPTO:1.0").is_none());
    }

    // ── InferProgress::apply ─────────────────────────────────────────

    #[test]
    fn progress_apply_slice_updates_current_and_total() {
        let mut p = InferProgress::default();
        p.apply(&InferMsg::Slice {
            current: 42,
            total: 155,
        });
        assert_eq!(p.current_slice, 42);
        assert_eq!(p.total_slices, 155);
    }

    #[test]
    fn progress_apply_volumes_accumulate() {
        let mut p = InferProgress::default();
        p.apply(&InferMsg::PartialVolume {
            class: 1,
            volume_ml: 2.3,
        });
        p.apply(&InferMsg::PartialVolume {
            class: 2,
            volume_ml: 8.7,
        });
        p.apply(&InferMsg::PartialVolume {
            class: 3,
            volume_ml: 1.5,
        });
        assert!((p.et_volume_ml - 2.3).abs() < 0.01);
        assert!((p.snfh_volume_ml - 8.7).abs() < 0.01);
        assert!((p.netc_volume_ml - 1.5).abs() < 0.01);
    }

    #[test]
    fn progress_apply_phase_updates_phase() {
        let mut p = InferProgress::default();
        assert_eq!(p.phase, InferPhase::Preprocessing);
        p.apply(&InferMsg::Phase(InferPhase::Slicing));
        assert_eq!(p.phase, InferPhase::Slicing);
        p.apply(&InferMsg::Phase(InferPhase::MarchingCubes));
        assert_eq!(p.phase, InferPhase::MarchingCubes);
    }

    #[test]
    fn progress_apply_done_is_noop() {
        let mut p = InferProgress::default();
        p.apply(&InferMsg::Done(true));
        // Fase nao muda com Done — Done e tratado pelo render loop, nao pelo progress
        assert_eq!(p.phase, InferPhase::Preprocessing);
    }

    #[test]
    fn progress_default_starts_at_preprocessing() {
        let p = InferProgress::default();
        assert_eq!(p.phase, InferPhase::Preprocessing);
        assert_eq!(p.current_slice, 0);
        assert_eq!(p.et_volume_ml, 0.0);
        assert_eq!(p.snfh_volume_ml, 0.0);
        assert_eq!(p.netc_volume_ml, 0.0);
    }
}
