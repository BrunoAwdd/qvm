use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Toffoli gate (CCNOT) — a universal 3-qubit logic gate.
///
/// It flips the third (target) qubit **only if** the first two (controls) are `1`.
///
/// Matrix representation (8×8):
/// - Identity except for the last two rows/cols (6 and 7) which are swapped:
///   - |110⟩ ↔ |111⟩
pub struct Toffoli {
    /// 8×8 matrix representing the CCNOT operation
    pub matrix: Array2<QLangComplex>,
}

impl Toffoli {
    /// Creates a new Toffoli gate (CCNOT).
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::<QLangComplex>::eye(8);

        // Swap |110⟩ and |111⟩ — positions 6 and 7
        matrix[[6, 6]] = zero;
        matrix[[7, 7]] = zero;
        matrix[[6, 7]] = one;
        matrix[[7, 6]] = one;

        Self { matrix }
    }
}

impl QuantumGateAbstract for Toffoli {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "Toffoli"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_toffoli_matrix_swap() {
        let toffoli = Toffoli::new();
        let m = toffoli.matrix;

        assert_eq!(m[[6, 7]], QLangComplex::one());
        assert_eq!(m[[7, 6]], QLangComplex::one());
        assert_eq!(m[[6, 6]], QLangComplex::zero());
        assert_eq!(m[[7, 7]], QLangComplex::zero());
    }

    #[test]
    fn test_toffoli_name() {
        let g = Toffoli::new();
        assert_eq!(g.name(), "Toffoli");
    }
}
