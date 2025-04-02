use ndarray::array;
use ndarray::Array2;

use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct RZ {
    pub matrix: Array2<CudaComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RZ {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "RZ"
    }
}

impl RZ {
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let phase_0 = CudaComplex::new((half_theta * -1.0).cos(), (half_theta * -1.0).sin());
        let phase_1 = CudaComplex::new((half_theta).cos(), (half_theta).sin());

        let matrix: Array2<CudaComplex> = array![
            [phase_0, CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), phase_1]
        ];

        Self { matrix, theta }
    }
}
