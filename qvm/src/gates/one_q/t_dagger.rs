use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::array;
use ndarray::Array2;

pub struct TDagger {
    pub matrix: Array2<QLangComplex>,
}

impl TDagger {
    pub fn new() -> Self {
        let phase = QLangComplex::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
        let matrix = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), phase],
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for TDagger {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "TDagger"
    }
}
