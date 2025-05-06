use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// RX(θ) gate — single-qubit rotation around the X-axis.
///
/// The matrix representation is:
///
/// ```text
/// RX(θ) = cos(θ/2) * I - i * sin(θ/2) * X
///
///        =
///        [ cos(θ/2)    -i·sin(θ/2) ]
///        [ -i·sin(θ/2)  cos(θ/2)   ]
/// ```
///
/// This gate rotates a qubit state around the X-axis of the Bloch sphere.
pub struct RX {
    /// The θ rotation angle in radians.
    pub theta: f64,

    /// The 2x2 matrix representation of RX(θ).
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for RX {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "RX"
    }
}

impl RX {
    /// Constructs a new RX(θ) gate for a given rotation angle `theta` (in radians).
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let cos = half_theta.cos();
        let sin = half_theta.sin();
        let i_sin = QLangComplex::new(0.0, -sin); // -i·sin(θ/2)

        let matrix = array![
            [QLangComplex::new(cos, 0.0), i_sin],
            [i_sin, QLangComplex::new(cos, 0.0)]
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
    fn test_rx_gate_matrix_pi_2() {
        let theta = FRAC_PI_2;
        let rx = RX::new(theta);

        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let expected = array![
            [QLangComplex::new(cos, 0.0), QLangComplex::new(0.0, -sin)],
            [QLangComplex::new(0.0, -sin), QLangComplex::new(cos, 0.0)]
        ];

        assert_eq!(rx.matrix, expected);
    }

    #[test]
    fn test_rx_gate_name() {
        let rx = RX::new(1.0);
        assert_eq!(rx.name(), "RX");
    }
}
