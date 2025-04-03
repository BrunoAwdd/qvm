// src/gates/identity.rs
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{array, Array2};

pub struct Identity {
    pub matrix: Array2<CudaComplex>,
}

impl Identity {
    pub fn new() -> Self {
        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0)],
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for Identity {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}
