use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;
use ndarray::array;
use ndarray::Array2;

pub struct U2 {
    pub phi: f64,
    pub lambda: f64,
    pub matrix: Array2<QLangComplex>,
}

impl U2 {
    pub fn new(phi: f64, lambda: f64) -> Self {
        let sqrt_2_inv = 1.0 / f64::sqrt(2.0);

        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);
        let e_i_phi = QLangComplex::from_polar(1.0, phi);
        let e_i_sum = QLangComplex::from_polar(1.0, phi + lambda);

        let matrix = array![
            [QLangComplex::new(sqrt_2_inv, 0.0), -e_i_lambda * sqrt_2_inv],
            [e_i_phi * sqrt_2_inv, e_i_sum * sqrt_2_inv]
        ];

        Self { phi, lambda, matrix }
    }
}

impl QuantumGateAbstract for U2 {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "U2"
    }
}
