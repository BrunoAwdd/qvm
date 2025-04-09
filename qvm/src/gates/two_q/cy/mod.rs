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
        let matrix = array![
            [
                QLangComplex::new(1.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0)
            ],
            [
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(1.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0)
            ],
            [
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, -1.0)
            ],
            [
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 1.0),
                QLangComplex::new(0.0, 0.0)
            ]
        ];
        Self { matrix }
    }
}
