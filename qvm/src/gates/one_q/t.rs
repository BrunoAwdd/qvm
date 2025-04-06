use ndarray::{array, Array2};
use std::f64::consts::PI;
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct T {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for T {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "T"
    }
}

impl T {
    pub fn new() -> Self {
        let angle = PI / 4.0;
        let phase = CudaComplex::new(angle.cos(), angle.sin()); // e^(iπ/4)

        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), phase]
        ];

        Self { matrix }
    }
}
