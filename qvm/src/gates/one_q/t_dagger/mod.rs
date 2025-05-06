use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{array, Array2};

/// The T† gate (T-dagger) — the inverse of the T gate.
///
/// It applies a −π/4 phase to the `|1⟩` state and is represented as:
///
/// ```text
/// | 1          0 |
/// | 0   e^{-iπ/4} |
/// ```
///
/// This gate is the Hermitian conjugate of the T (π/8) gate and is useful
/// in reversible and error-corrected quantum circuits.
pub struct TDagger {
    /// The 2x2 matrix representation of the T† gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for TDagger {
    /// Returns the matrix of the T† gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the gate's name: `"TDagger"`.
    fn name(&self) -> &'static str {
        "TDagger"
    }
}

impl TDagger {
    /// Constructs a new T† gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        // e^{-iπ/4} = cos(-π/4) + i·sin(-π/4)
        let phase = QLangComplex::from_polar(1.0, -std::f64::consts::FRAC_PI_4);

        let matrix = array![
            [one, zero],
            [zero, phase],
        ];

        Self { matrix }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn test_t_dagger_matrix() {
        let t_dagger = TDagger::new();
        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::from_polar(1.0, -FRAC_PI_4)]
        ];
        assert_eq!(t_dagger.matrix, expected);
    }

    #[test]
    fn test_t_dagger_name() {
        let t_dagger = TDagger::new();
        assert_eq!(t_dagger.name(), "TDagger");
    }
}
