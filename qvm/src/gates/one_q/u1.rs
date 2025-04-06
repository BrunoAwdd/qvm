use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::qvm::cuda::types::CudaComplex;
use ndarray::array;
use ndarray::Array2;

pub struct U1 {
    pub lambda: f64,
    pub matrix: Array2<CudaComplex>,
}

impl U1 {
    pub fn new(lambda: f64) -> Self {
        let e_i_lambda = CudaComplex::from_polar(1.0, lambda);

        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), e_i_lambda]
        ];

        Self { lambda, matrix }
    }
}

impl QuantumGateAbstract for U1 {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "U1"
    }
}
