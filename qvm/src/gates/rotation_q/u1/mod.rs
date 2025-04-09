use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;
use ndarray::array;
use ndarray::Array2;

pub struct U1 {
    pub lambda: f64,
    pub matrix: Array2<QLangComplex>,
}

impl U1 {
    pub fn new(lambda: f64) -> Self {
        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);

        let matrix = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), e_i_lambda]
        ];

        Self { lambda, matrix }
    }
}

impl QuantumGateAbstract for U1 {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "U1"
    }
}
