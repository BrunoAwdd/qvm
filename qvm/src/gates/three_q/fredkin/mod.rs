use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Fredkin gate (Controlled-SWAP).
///
/// This is a 3-qubit gate that swaps the last two qubits only when the first is `1`.
///
/// Matrix representation (8×8):
/// - Identity except for rows/cols 5 and 6, which are swapped:
///   - |101⟩ ↔ |110⟩
///
/// Gate diagram:
/// ```text
/// control ─●────
///          │
/// target ──X────
///          │
/// target ──X────
/// ```
pub struct Fredkin {
    /// The 8×8 unitary matrix representing the gate
    pub matrix: Array2<QLangComplex>,
}

impl Fredkin {
    /// Constructs a new Fredkin (CSWAP) gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::<QLangComplex>::eye(8);

        // Swap positions 5 and 6 (|101⟩ ↔ |110⟩)
        matrix[[5, 5]] = zero;
        matrix[[6, 6]] = zero;
        matrix[[5, 6]] = one;
        matrix[[6, 5]] = one;

        Self { matrix }
    }
}

impl QuantumGateAbstract for Fredkin {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "Fredkin"
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_fredkin_matrix_swap() {
        let fredkin = Fredkin::new();
        let m = fredkin.matrix;

        assert_eq!(m[[5, 6]], QLangComplex::one());
        assert_eq!(m[[6, 5]], QLangComplex::one());
        assert_eq!(m[[5, 5]], QLangComplex::zero());
        assert_eq!(m[[6, 6]], QLangComplex::zero());
    }

    #[test]
    fn test_fredkin_name() {
        let g = Fredkin::new();
        assert_eq!(g.name(), "Fredkin");
    }
}
