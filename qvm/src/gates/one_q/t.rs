use ndarray::{array, Array2};
use std::f64::consts::PI;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct T {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for T {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "T"
    }
}

impl T {
    pub fn new() -> Self {
        let angle = PI / 4.0;
        let phase = QLangComplex::new(angle.cos(), angle.sin()); // e^(iπ/4)

        let matrix = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), phase]
        ];

        Self { matrix }
    }
}
