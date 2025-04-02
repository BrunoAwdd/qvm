
use ndarray::array;
use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::Array2;

pub struct U3 {
    pub theta: f64,
    pub phi: f64,
    pub lambda: f64,
    pub matrix: Array2<CudaComplex>,
}

impl QuantumGateAbstract for U3 {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "U3"
    }

    fn as_u3_params(&self) -> Option<(f64, f64, f64)> {
        Some((self.theta, self.phi, self.lambda))
    }
}

impl U3 {
    pub fn new(theta: f64, phi: f64, lambda: f64) -> Self {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let e_i_phi = CudaComplex::from_polar(1.0, phi);
        let e_i_lambda = CudaComplex::from_polar(1.0, lambda);
        let e_i_sum = CudaComplex::from_polar(1.0, phi + lambda);

        let matrix = array![
            [CudaComplex::new(cos, 0.0), -e_i_lambda * sin],
            [e_i_phi * sin, e_i_sum * cos]
        ];

        Self { theta, phi, lambda, matrix }
    }
}