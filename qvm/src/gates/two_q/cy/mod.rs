use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct ControlledY {
    matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for ControlledY {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "cy"
    }
}

impl ControlledY {
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();
        let neg_i = QLangComplex::neg_i();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, neg_i],
            [zero, zero, i, zero]
        ];
        Self { matrix }
    }
}
