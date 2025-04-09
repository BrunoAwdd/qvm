use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{Array2, array};

pub struct SDagger {
    pub matrix: Array2<QLangComplex>,
}

impl SDagger {
    pub fn new() -> Self {
        let matrix = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, -1.0)]
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for SDagger {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "SDagger"
    }
}
