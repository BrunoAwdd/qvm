use ndarray::array;
use ndarray::Array2;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct RY {
    pub matrix: Array2<QLangComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RY {
    fn matrix(&self) -> Array2<QLangComplex> {
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

        let matrix: Array2<QLangComplex> = array![
            [QLangComplex::new(cos.cos(), 0.0), QLangComplex::new(-sin.sin(), 0.0)],
            [QLangComplex::new(sin.sin(), 0.0), QLangComplex::new(cos.cos(), 0.0)]
        ];

        Self { matrix, theta }
    }
}
