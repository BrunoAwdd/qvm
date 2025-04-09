use ndarray::array;
use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Phase {
    pub theta: f64,
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Phase {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "phase"
    }
}

impl Phase {
    pub fn new(theta: f64) -> Self {
        let phase = QLangComplex::from_polar(1.0, theta); // e^(iθ)

        let matrix = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), phase]
        ];

        Self { theta, matrix }
    }
}
