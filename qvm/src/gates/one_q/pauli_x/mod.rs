use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// The Pauli-X gate (also known as the X gate or quantum NOT gate).
///
/// It flips the state of a single qubit:
/// - `|0⟩` becomes `|1⟩`
/// - `|1⟩` becomes `|0⟩`
///
/// The matrix representation is:
///
/// ```text
/// | 0  1 |
/// | 1  0 |
/// ```
pub struct PauliX {
    /// The 2x2 matrix representation of the Pauli-X gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for PauliX {
    /// Returns the matrix of the Pauli-X gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the name of the gate.
    fn name(&self) -> &'static str {
        "PauliX"
    }
}

impl PauliX {
    /// Constructs a new Pauli-X gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [zero, one],
            [one, zero]
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
    fn test_pauli_x_matrix() {
        let px = PauliX::new();
        let expected = array![
            [QLangComplex::zero(), QLangComplex::one()],
            [QLangComplex::one(), QLangComplex::zero()]
        ];

        assert_eq!(px.matrix, expected);
    }

    #[test]
    fn test_pauli_x_name() {
        let px = PauliX::new();
        assert_eq!(px.name(), "PauliX");
    }
}
