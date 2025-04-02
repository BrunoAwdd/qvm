use ndarray::{array, Array2};
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Fredkin {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for Fredkin {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Fredkin"
    }
}

impl Fredkin {
    pub fn new() -> Self {
        let mut mat = Array2::<CudaComplex>::eye(8);
        mat[[5, 5]] = CudaComplex::new(0.0, 0.0);
        mat[[6, 6]] = CudaComplex::new(0.0, 0.0);
        mat[[5, 6]] = CudaComplex::new(1.0, 0.0);
        mat[[6, 5]] = CudaComplex::new(1.0, 0.0);
        Self { matrix: mat }
    }
}
