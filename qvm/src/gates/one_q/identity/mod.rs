// src/gates/identity.rs
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{array, Array2};

/// The Identity gate (I) — a no-op quantum gate.
///
/// This gate leaves a qubit unchanged. It acts as the 2x2 identity matrix:
///
/// ```text
/// | 1  0 |
/// | 0  1 |
/// ```
///
/// It is often used in composite gates or for timing/synchronization purposes.
pub struct Identity {
    /// The 2x2 matrix representation of the Identity gate.
    pub matrix: Array2<QLangComplex>,
}

impl Identity {
    /// Constructs a new Identity gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, one],
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for Identity {
    /// Returns the identity matrix.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the gate name.
    fn name(&self) -> &'static str {
        "Identity"
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;

    #[test]
    fn test_identity_matrix_correctness() {
        let id = Identity::new();
        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::one()]
        ];
        assert_eq!(id.matrix, expected);
    }

    #[test]
    fn test_identity_name() {
        let id = Identity::new();
        assert_eq!(id.name(), "Identity");
    }
}
