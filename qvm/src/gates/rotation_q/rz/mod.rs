use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// RZ(θ) gate — single-qubit rotation around the Z-axis.
///
/// Matrix representation:
/// ```text
/// RZ(θ) =
/// [ e^{-iθ/2}     0     ]
/// [    0      e^{iθ/2}  ]
/// ```
///
/// This gate applies opposite phase shifts to the `|0⟩` and `|1⟩` basis states.
pub struct RZ {
    /// Rotation angle in radians
    pub theta: f64,

    /// 2x2 unitary matrix for RZ(θ)
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for RZ {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "RZ"
    }
}

impl RZ {
    /// Creates a new RZ(θ) gate
    pub fn new(theta: f64) -> Self {
        let half_theta = theta / 2.0;
        let phase_0 = QLangComplex::from_polar(1.0, -half_theta);
        let phase_1 = QLangComplex::from_polar(1.0, half_theta);

        let zero = QLangComplex::zero();

        let matrix = array![
            [phase_0, zero],
            [zero, phase_1]
        ];

        Self { matrix, theta }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::types::qlang_complex::QLangComplex;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_rz_matrix() {
        let theta = FRAC_PI_2;
        let rz = RZ::new(theta);
        let half = theta / 2.0;

        let expected = array![
            [QLangComplex::from_polar(1.0, -half), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::from_polar(1.0, half)]
        ];

        assert_eq!(rz.matrix, expected);
    }

    #[test]
    fn test_rz_name() {
        let rz = RZ::new(1.0);
        assert_eq!(rz.name(), "RZ");
    }
}

