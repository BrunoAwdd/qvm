use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Controlled-Z (CZ) gate.
///
/// This 2-qubit gate applies a Pauli-Z to the target qubit **only when**
/// the control qubit is in the `|1⟩` state.
///
/// Matrix representation (4×4):
/// ```text
/// [ 1  0  0   0 ]
/// [ 0  1  0   0 ]
/// [ 0  0  1   0 ]
/// [ 0  0  0  -1 ]
/// ```
///
/// It is a symmetric gate (CZ = ZC).
pub struct ControlledZ {
    /// The 4×4 unitary matrix for the CZ gate
    matrix: Array2<QLangComplex>,
}

impl ControlledZ {
    /// Creates a new Controlled-Z gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_one = QLangComplex::neg_one();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, neg_one],
        ];

        Self { matrix }
    }
}

impl QuantumGateAbstract for ControlledZ {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "cz"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_cz_matrix() {
        let cz = ControlledZ::new();
        let m = cz.matrix();

        assert_eq!(m[[0, 0]], QLangComplex::one());
        assert_eq!(m[[1, 1]], QLangComplex::one());
        assert_eq!(m[[2, 2]], QLangComplex::one());
        assert_eq!(m[[3, 3]], QLangComplex::neg_one());
    }

    #[test]
    fn test_cz_name() {
        let cz = ControlledZ::new();
        assert_eq!(cz.name(), "cz");
    }
}
