use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{Array2, array};

/// The S-dagger gate (S†) — the inverse of the S (phase) gate.
///
/// The matrix representation is:
/// ```text
/// | 1    0 |
/// | 0  -i |
/// ```
///
/// It reverses the π/2 phase shift applied by the S gate.
/// S† * S = Identity.
pub struct SDagger {
    /// The 2x2 matrix representation of the S† gate.
    pub matrix: Array2<QLangComplex>,
}

impl SDagger {
    /// Constructs a new S† gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_i = QLangComplex::neg_i();

        let matrix = array![
            [one, zero],
            [zero, neg_i]
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for SDagger {
    /// Returns the matrix of the S† gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the name of the gate.
    fn name(&self) -> &'static str {
        "SDagger"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;
    use ndarray::array;

    #[test]
    fn test_s_dagger_matrix() {
        let s_dagger = SDagger::new();
        let expected = array![
            [QLangComplex::one(), QLangComplex::zero()],
            [QLangComplex::zero(), QLangComplex::neg_i()]
        ];
        assert_eq!(s_dagger.matrix, expected);
    }

    #[test]
    fn test_s_dagger_name() {
        let s_dagger = SDagger::new();
        assert_eq!(s_dagger.name(), "SDagger");
    }
}
