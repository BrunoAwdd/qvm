use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;
use ndarray::array;
use ndarray::Array2;

pub struct U1 {
    pub lambda: f64,
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for U1 {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "U1"
    }
}

impl U1 {
    pub fn new(lambda: f64) -> Self {
        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, e_i_lambda]
        ];

        Self { lambda, matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_u1_matrix() {
        let lambda = PI;
        let u1 = U1::new(lambda);
        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::from_polar(1.0, lambda)]
        ];
        assert_eq!(u1.matrix, expected);
    }

    #[test]
    fn test_u1_name() {
        let u1 = U1::new(1.0);
        assert_eq!(u1.name(), "U1");
    }
}

