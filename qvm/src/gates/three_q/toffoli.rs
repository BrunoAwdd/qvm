use ndarray::{array, Array2};
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Toffoli {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for Toffoli {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Toffoli"
    }
}

impl Toffoli {
    pub fn new() -> Self {
        let mut mat = Array2::<CudaComplex>::eye(8);
        mat[[6, 6]] = CudaComplex::new(0.0, 0.0);
        mat[[7, 7]] = CudaComplex::new(0.0, 0.0);
        mat[[6, 7]] = CudaComplex::new(1.0, 0.0);
        mat[[7, 6]] = CudaComplex::new(1.0, 0.0);
        Self { matrix: mat }
    }
}
