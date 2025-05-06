use ndarray::{array, Array2};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// The S gate — also known as the phase gate or √Z.
///
/// The S gate applies a π/2 phase shift to the `|1⟩` basis state:
///
/// ```text
/// | 1   0 |
/// | 0   i |
/// ```
///
/// It is equivalent to the square root of the Pauli-Z gate (Z), and satisfies:
/// - `S * S = Z`
/// - `S† = S⁻¹ = S†` (its inverse is the S-dagger gate)
pub struct S {
    /// The 2x2 matrix representation of the S gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for S {
    /// Returns the matrix representation of the S gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the gate name: `"S"`.
    fn name(&self) -> &'static str {
        "S"
    }
}

impl S {
    /// Constructs a new S gate.
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();

        let matrix = array![
            [one, zero],
            [zero, i]
        ];
        Self { matrix }
    }
}
