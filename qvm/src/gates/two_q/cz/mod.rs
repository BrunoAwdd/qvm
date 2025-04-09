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
                QLangComplex::new(1.0, 0.0),
                QLangComplex::new(0.0, 0.0)
            ],
            [
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(0.0, 0.0),
                QLangComplex::new(-1.0, 0.0)
            ]
        ];
        Self { matrix }
    }
}
