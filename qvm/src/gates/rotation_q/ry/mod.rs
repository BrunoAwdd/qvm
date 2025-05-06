use ndarray::array;
use ndarray::Array2;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// RY(θ) gate — single-qubit rotation around the Y-axis.
///
/// Matrix representation:
///
/// ```text
/// RY(θ) =
/// [  cos(θ/2)   -sin(θ/2) ]
/// [  sin(θ/2)    cos(θ/2) ]
/// ```
///
/// This gate rotates a qubit around the Y axis of the Bloch sphere.
pub struct RY {
    pub matrix: Array2<QLangComplex>,
    pub theta: f64,
}

impl QuantumGateAbstract for RY {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "RY"
    }
}

impl RY {
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let cos = half_theta.cos();
        let sin = half_theta.sin();

        let matrix = array![
            [QLangComplex::new(cos, 0.0), QLangComplex::new(-sin, 0.0)],
            [QLangComplex::new(sin, 0.0), QLangComplex::new(cos, 0.0)],
        ];

        Self { matrix, theta }
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_ry_matrix() {
        let theta = FRAC_PI_2;
        let ry = RY::new(theta);
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let expected = array![
            [QLangComplex::new(cos, 0.0), QLangComplex::new(-sin, 0.0)],
            [QLangComplex::new(sin, 0.0), QLangComplex::new(cos, 0.0)],
        ];

        assert_eq!(ry.matrix, expected);
    }

    #[test]
    fn test_ry_name() {
        let ry = RY::new(1.0);
        assert_eq!(ry.name(), "RY");
    }
}
