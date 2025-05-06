use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// iSWAP gate — swaps |01⟩ and |10⟩ with a phase of i.
///
/// Matrix form (4×4):
/// ```text
/// [ 1  0   0   0 ]
/// [ 0  0   i   0 ]
/// [ 0  i   0   0 ]
/// [ 0  0   0   1 ]
/// ```
///
/// - |01⟩ → i|10⟩
/// - |10⟩ → i|01⟩
pub struct ISwap {
    /// 4×4 matrix representing the gate
    matrix: Array2<QLangComplex>,
}

impl ISwap {
    /// Constructs a new iSWAP gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();

        let matrix = array![
            [one,  zero, zero, zero],
            [zero, zero,   i,  zero],
            [zero,   i,  zero, zero],
            [zero, zero, zero, one],
        ];

        Self { matrix }
    }
}

impl QuantumGateAbstract for ISwap {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "iswap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_iswap_matrix() {
        let g = ISwap::new();
        let m = g.matrix();
        let i = QLangComplex::i();
        let one = QLangComplex::one();
        let zero = QLangComplex::zero();

        let expected = array![
            [one, zero, zero, zero],
            [zero, zero, i, zero],
            [zero, i, zero, zero],
            [zero, zero, zero, one],
        ];

        assert_eq!(m, expected);
    }

    #[test]
    fn test_iswap_name() {
        let g = ISwap::new();
        assert_eq!(g.name(), "iswap");
    }
}
