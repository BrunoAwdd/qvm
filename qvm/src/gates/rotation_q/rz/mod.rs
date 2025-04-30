use ndarray::array;
use ndarray::Array2;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct RZ {
    pub matrix: Array2<QLangComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RZ {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "RZ"
    }
}

impl RZ {
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let phase_0 = QLangComplex::new((half_theta * -1.0).cos(), (half_theta * -1.0).sin());
        let phase_1 = QLangComplex::new((half_theta).cos(), (half_theta).sin());

        let zero = QLangComplex::zero();

        let matrix: Array2<QLangComplex> = array![
            [phase_0, zero],
            [zero, phase_1]
        ];

        Self { matrix, theta }
    }
}
