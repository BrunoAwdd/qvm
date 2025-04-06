
use ndarray::array;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::Array2;

pub struct U3 {
    pub theta: f64,
    pub phi: f64,
    pub lambda: f64,
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for U3 {
    fn matrix(&self) -> Array2<QLangComplex> {
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

        let e_i_phi = QLangComplex::from_polar(1.0, phi);
        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);
        let e_i_sum = QLangComplex::from_polar(1.0, phi + lambda);

        let matrix = array![
            [QLangComplex::new(cos, 0.0), -e_i_lambda * sin],
            [e_i_phi * sin, e_i_sum * cos]
        ];

        Self { theta, phi, lambda, matrix }
    }
}