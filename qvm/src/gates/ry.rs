use ndarray::array;
use ndarray::Array2;

use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct RY {
    pub matrix: Array2<CudaComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RY {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "RY"
    }
}

impl RY {
    pub fn new(theta: f64) -> Self {
        let cos = theta / 2.0;
        let sin = cos;

        let matrix: Array2<CudaComplex> = array![
            [CudaComplex::new(cos.cos(), 0.0), CudaComplex::new(-sin.sin(), 0.0)],
            [CudaComplex::new(sin.sin(), 0.0), CudaComplex::new(cos.cos(), 0.0)]
        ];

        Self { matrix, theta }
    }
}
