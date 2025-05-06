use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// SWAP gate — swaps the states of two qubits.
///
/// Matrix form:
/// ```text
/// [ 1  0  0  0 ]
/// [ 0  0  1  0 ]
/// [ 0  1  0  0 ]
/// [ 0  0  0  1 ]
/// ```
///
/// Effect:
/// - |00⟩ → |00⟩  
/// - |01⟩ → |10⟩  
/// - |10⟩ → |01⟩  
/// - |11⟩ → |11⟩
pub struct Swap {
    pub matrix: Array2<QLangComplex>,
}

impl Swap {
    /// Creates a new SWAP gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero, zero, zero],
            [zero, zero, one, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
        ];

        Self { matrix }
    }
}

impl QuantumGateAbstract for Swap {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "SWAP"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_swap_matrix() {
        let swap = Swap::new();
        let m = swap.matrix();

        let one = QLangComplex::one();
        let zero = QLangComplex::zero();

        let expected = array![
            [one, zero, zero, zero],
            [zero, zero, one, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
        ];

        assert_eq!(m, expected);
    }

    #[test]
    fn test_swap_name() {
        let g = Swap::new();
        assert_eq!(g.name(), "SWAP");
    }
}
