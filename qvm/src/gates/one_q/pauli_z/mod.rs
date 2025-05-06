use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// The Pauli-Z gate — a fundamental single-qubit quantum gate.
///
/// It leaves `|0⟩` unchanged and flips the phase of `|1⟩`:
///
/// ```text
/// | 1   0 |
/// | 0  -1 |
/// ```
///
/// This gate is often used to perform conditional phase flips and is
/// part of the standard Pauli set {X, Y, Z}.
pub struct PauliZ {
    /// The 2x2 matrix representation of the Pauli-Z gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for PauliZ {
    /// Returns the matrix representation of the Pauli-Z gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the name of the gate.
    fn name(&self) -> &'static str {
        "PauliZ"
    }
}

impl PauliZ {
    /// Constructs a new Pauli-Z gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_one = QLangComplex::neg_one();

        let matrix = array![
            [one, zero],
            [zero, neg_one]
        ];

        Self { matrix }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;

    #[test]
    fn test_pauli_z_matrix() {
        let pz = PauliZ::new();
        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::neg_one()]
        ];
        assert_eq!(pz.matrix, expected);
    }

    #[test]
    fn test_pauli_z_name() {
        let pz = PauliZ::new();
        assert_eq!(pz.name(), "PauliZ");
    }
}
