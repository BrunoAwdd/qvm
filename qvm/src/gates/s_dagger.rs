use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{Array2, array};

pub struct SDagger {
    matrix: Array2<CudaComplex>,
}

impl SDagger {
    pub fn new() -> Self {
        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, -1.0)]
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for SDagger {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "SDagger"
    }
}
