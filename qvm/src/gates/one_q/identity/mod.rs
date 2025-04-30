// src/gates/identity.rs
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{array, Array2};

pub struct Identity {
    pub matrix: Array2<QLangComplex>,
}

impl Identity {
    pub fn new() -> Self {

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, one],
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for Identity {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}
