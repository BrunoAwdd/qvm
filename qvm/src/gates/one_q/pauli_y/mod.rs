use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// The Pauli-Y gate — a fundamental single-qubit quantum gate.
///
/// It is defined by the matrix:
/// ```text
/// |  0  -i |
/// |  i   0 |
/// ```
///
/// It performs a bit flip (like Pauli-X) combined with a phase flip.
/// When applied to `|0⟩`, it produces `i|1⟩`; and to `|1⟩`, `-i|0⟩`.
pub struct PauliY {
    /// The 2x2 matrix representation of the Pauli-Y gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for PauliY {
    /// Returns the matrix representation of the Pauli-Y gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the gate's name.
    fn name(&self) -> &'static str {
        "pauliY"
    }
}

impl PauliY {
    /// Constructs a new Pauli-Y gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let i = QLangComplex::i();       // i = (0, 1)
        let neg_i = QLangComplex::neg_i(); // -i = (0, -1)

        let matrix = array![
            [zero, neg_i],
            [i, zero]
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
    fn test_pauli_y_matrix() {
        let py = PauliY::new();
        let expected = array![
            [QLangComplex::zero(), QLangComplex::neg_i()],
            [QLangComplex::i(), QLangComplex::zero()]
        ];

        assert_eq!(py.matrix, expected);
    }

    #[test]
    fn test_pauli_y_name() {
        let py = PauliY::new();
        assert_eq!(py.name(), "pauliY");
    }
}
