use ndarray::array;
use ndarray::Array2;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct RX {
    pub matrix: Array2<QLangComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RX {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "RX"
    }
}

impl RX {
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let cos = half_theta.cos();
        let sin = half_theta.sin();
        let i_sin = QLangComplex::new(0.0, -sin); // -i * sin(θ/2)

        let matrix: Array2<QLangComplex> = array![
            [QLangComplex::new(cos, 0.0), i_sin],
            [i_sin, QLangComplex::new(cos, 0.0)]
        ];

        Self { matrix, theta }
    }
}
