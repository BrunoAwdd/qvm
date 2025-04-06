
use ndarray::Array2;
use ndarray::array;

use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;


pub struct Hadamard {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for Hadamard {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Hadamard"
    }
}

impl Hadamard {
    pub fn new() -> Self {
        let factor: f64 = 1.0 / (2.0_f64).sqrt();
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<CudaComplex>, ndarray::Dim<[usize; 2]>> = array![
            [CudaComplex::new(factor, 0.0), CudaComplex::new(factor, 0.0)],
            [CudaComplex::new(factor, 0.0), CudaComplex::new(-factor, 0.0)]
        ];

        Self { matrix }
    }
}

