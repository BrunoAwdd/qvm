use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// A generic Phase gate with an arbitrary angle θ.
///
/// The matrix is defined as:
/// ```text
/// | 1       0 |
/// | 0   e^{iθ} |
/// ```
///
/// Special cases:
/// - θ = π/2 → S gate
/// - θ = π/4 → T gate
/// - θ = -π/4 → T†
///
/// This gate introduces a controllable phase shift to the `|1⟩` state.
pub struct Phase {
    /// The angle θ in radians.
    pub theta: f64,

    /// The 2x2 matrix representation of the Phase gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Phase {
    /// Returns the matrix representation of the Phase gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the name of the gate: `"phase"`.
    fn name(&self) -> &'static str {
        "phase"
    }
}

impl Phase {
    /// Constructs a new Phase gate with the given angle θ (in radians).
    pub fn new(theta: f64) -> Self {
        let phase = QLangComplex::from_polar(1.0, theta); // e^(iθ)

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, phase]
        ];

        Self { theta, matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_phase_gate_matrix_theta_pi_2() {
        let theta = PI / 2.0;
        let phase = Phase::new(theta);

        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::from_polar(1.0, theta)]
        ];

        assert_eq!(phase.matrix, expected);
    }

    #[test]
    fn test_phase_gate_name() {
        let p = Phase::new(0.0);
        assert_eq!(p.name(), "phase");
    }
}
