use ndarray::{array, Array2};
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct S {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for S {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "S"
    }
}

impl S {
    pub fn new() -> Self {
        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 1.0)]
        ];
        Self { matrix }
    }
}
