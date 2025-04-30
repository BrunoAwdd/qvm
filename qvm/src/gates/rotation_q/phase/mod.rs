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

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, phase]
        ];

        Self { theta, matrix }
    }
}
