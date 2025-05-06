use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Controlled-Y (CY) gate — applies Y to the target qubit if the control is 1.
///
/// Matrix representation (4×4):
/// ```text
/// [ 1  0   0     0  ]
/// [ 0  1   0     0  ]
/// [ 0  0   0   -i  ]
/// [ 0  0   i     0  ]
/// ```
///
/// - Acts like identity when control = 0
/// - Applies Pauli-Y when control = 1
pub struct ControlledY {
    /// The 4×4 unitary matrix for CY
    matrix: Array2<QLangComplex>,
}

impl ControlledY {
    /// Creates a new Controlled-Y gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();
        let neg_i = QLangComplex::neg_i();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, neg_i],
            [zero, zero, i, zero],
        ];

        Self { matrix }
    }
}

impl QuantumGateAbstract for ControlledY {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "cy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_cy_matrix_structure() {
        let cy = ControlledY::new();
        let m = cy.matrix();

        let i = QLangComplex::i();
        let neg_i = QLangComplex::neg_i();
        let one = QLangComplex::one();
        let zero = QLangComplex::zero();

        let expected = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, neg_i],
            [zero, zero, i, zero],
        ];

        assert_eq!(m, expected);
    }

    #[test]
    fn test_cy_name() {
        let cy = ControlledY::new();
        assert_eq!(cy.name(), "cy");
    }
}

