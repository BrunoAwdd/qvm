use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// U3(θ, φ, λ) gate — the most general single-qubit unitary gate.
///
/// Matrix representation:
/// ```text
/// U3(θ, φ, λ) =
/// [  cos(θ/2)                -e^{iλ}·sin(θ/2) ]
/// [  e^{iφ}·sin(θ/2)   e^{i(φ+λ)}·cos(θ/2) ]
/// ```
///
/// - U3(θ, φ, λ) is universal for single-qubit operations.
/// - U2 and U1 are special cases:
///   - U2(φ, λ) = U3(π/2, φ, λ)
///   - U1(λ)    = U3(0, 0, λ)
pub struct U3 {
    /// Rotation angle θ
    pub theta: f64,
    /// Phase φ
    pub phi: f64,
    /// Phase λ
    pub lambda: f64,
    /// The 2×2 unitary matrix
    pub matrix: Array2<QLangComplex>,
}

impl U3 {
    /// Constructs a new U3 gate from the given parameters.
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_u3_matrix() {
        let theta = PI / 2.0;
        let phi = PI / 4.0;
        let lambda = PI / 6.0;

        let u3 = U3::new(theta, phi, lambda);
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let e_i_phi = QLangComplex::from_polar(1.0, phi);
        let e_i_lambda = QLangComplex::from_polar(1.0, lambda);
        let e_i_sum = QLangComplex::from_polar(1.0, phi + lambda);

        let expected = array![
            [QLangComplex::new(cos, 0.0), -e_i_lambda * sin],
            [e_i_phi * sin, e_i_sum * cos]
        ];

        assert_eq!(u3.matrix, expected);
    }

    #[test]
    fn test_u3_name() {
        let u3 = U3::new(0.0, 0.0, 0.0);
        assert_eq!(u3.name(), "U3");
    }

    #[test]
    fn test_u3_params() {
        let u3 = U3::new(1.0, 2.0, 3.0);
        assert_eq!(u3.as_u3_params(), Some((1.0, 2.0, 3.0)));
    }
}
