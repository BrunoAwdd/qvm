use crate::qvm::cuda::types::CudaComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::array;
use ndarray::Array2;

pub struct TDagger {
    pub matrix: Array2<CudaComplex>,
}

impl TDagger {
    pub fn new() -> Self {
        let phase = CudaComplex::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
        let matrix = array![
            [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
            [CudaComplex::new(0.0, 0.0), phase],
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for TDagger {
    fn matrix(&self) -> Array2<CudaComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "TDagger"
    }
}
