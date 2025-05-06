use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;
use ndarray::{array, Array2};

/// U2(φ, λ) gate — a single-qubit rotation parameterized by phase shifts.
///
/// This gate is defined as:
/// ```text
/// U2(φ, λ) = 1/√2 × [  1           -e^{iλ}       ]
///                        [  e^{iφ}   e^{i(φ+λ)}   ]
/// ```
///
/// This gate is equivalent to a U3 gate with θ = π/2, and is widely used
/// in Qiskit's standard decomposition for arbitrary single-qubit operations.
pub struct U2 {
    /// First phase parameter φ (radians)
    pub phi: f64,

    /// Second phase parameter λ (radians)
    pub lambda: f64,

    /// The 2x2 matrix representation of U2(φ, λ)
    pub matrix: Array2<QLangComplex>,
}

impl U2 {
    /// Constructs a new U2 gate with given phase parameters φ and λ.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::types::qlang_complex::QLangComplex;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    #[test]
    fn test_u2_matrix() {
        let phi = PI / 2.0;
        let lambda = PI / 4.0;
        let u2 = U2::new(phi, lambda);

        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);
        let e_i_phi = QLangComplex::from_polar(1.0, phi);
        let e_i_sum = QLangComplex::from_polar(1.0, phi + lambda);

        let expected = array![
            [QLangComplex::new(FRAC_1_SQRT_2, 0.0), -e_i_lambda * FRAC_1_SQRT_2],
            [e_i_phi * FRAC_1_SQRT_2, e_i_sum * FRAC_1_SQRT_2]
        ];

        assert_eq!(u2.matrix, expected);
    }

    #[test]
    fn test_u2_name() {
        let u2 = U2::new(0.0, 0.0);
        assert_eq!(u2.name(), "U2");
    }
}
