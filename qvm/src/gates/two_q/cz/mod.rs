use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct ControlledZ {
    matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for ControlledZ {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "cz"
    }
}

impl ControlledZ {
    pub fn new() -> Self {

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_one = QLangComplex::neg_one();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, neg_one]
        ];
        Self { matrix }
    }
}
