use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// CNOT gate — Controlled-NOT (CX) gate.
///
/// This 2-qubit gate flips the target qubit if the control qubit is `1`.
///
/// Matrix form (4×4):
/// ```text
/// [ 1 0 0 0 ]
/// [ 0 1 0 0 ]
/// [ 0 0 0 1 ]
/// [ 0 0 1 0 ]
/// ```
///
/// - |00⟩ → |00⟩  
/// - |01⟩ → |01⟩  
/// - |10⟩ → |11⟩  
/// - |11⟩ → |10⟩
pub struct CNOT {
    /// The 4×4 unitary matrix for the gate
    pub matrix: Array2<QLangComplex>,
}

impl CNOT {
    /// Creates a new CNOT (CX) gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, one, zero],
        ];

        Self { matrix }
    }
}

impl QuantumGateAbstract for CNOT {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "CNOT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_cnot_matrix() {
        let cnot = CNOT::new();
        let m = cnot.matrix;

        assert_eq!(m[[0, 0]], QLangComplex::one());
        assert_eq!(m[[1, 1]], QLangComplex::one());
        assert_eq!(m[[2, 3]], QLangComplex::one());
        assert_eq!(m[[3, 2]], QLangComplex::one());

        // Ensure swapped entries
        assert_eq!(m[[2, 2]], QLangComplex::zero());
        assert_eq!(m[[3, 3]], QLangComplex::zero());
    }

    #[test]
    fn test_cnot_name() {
        let cnot = CNOT::new();
        assert_eq!(cnot.name(), "CNOT");
    }
}
