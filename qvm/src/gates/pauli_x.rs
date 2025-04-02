use ndarray::array;
use ndarray::Array2;
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct PauliX {
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for PauliX {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "PauliX"
    }
}

impl PauliX {
    pub fn new() -> Self {
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<CudaComplex>, ndarray::Dim<[usize; 2]>> = array![
            [CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0)],
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)]
        ];

        Self { matrix }
    }
}
