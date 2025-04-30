use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct S {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for S {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "S"
    }
}

impl S {
    pub fn new() -> Self {

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();

        let matrix = array![
            [one, zero],
            [zero, i]
        ];
        Self { matrix }
    }
}
