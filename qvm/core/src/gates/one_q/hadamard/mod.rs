use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;
use ndarray::{array, Array2};

/// The Hadamard gate (H) — a one-qubit gate that creates superposition.
///
/// The Hadamard matrix is:
/// ```text
/// 1/sqrt(2) * |  1   1 |
///             |  1  -1 |
/// ```
///
/// When applied to |0⟩, it produces (|0⟩ + |1⟩)/√2.
/// When applied to |1⟩, it produces (|0⟩ - |1⟩)/√2.
pub struct Hadamard {
    /// The 2x2 matrix representing the Hadamard gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Hadamard {
    /// Returns the matrix of the Hadamard gate.
    fn matrix(&self) -> Array2<QLangComplex> { self.matrix.clone() }

    /// Returns the name of the gate.
    fn name(&self) -> &'static str { "Hadamard" }
}

impl Hadamard {
    /// Constructs a new Hadamard gate.
    pub fn new() -> Self {
        let factor: f64 = 1.0 / (2.0_f64).sqrt();
        let matrix = array![
            [
                QLangComplex::new(factor, 0.0),
                QLangComplex::new(factor, 0.0)
            ],
            [
                QLangComplex::new(factor, 0.0),
                QLangComplex::new(-factor, 0.0)
            ]
        ];

        Self { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::one_q::hadamard::Hadamard;

    use ndarray::array;

    #[test]
    fn test_hadamard_matrix() {
        let h = Hadamard::new();
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();

        let expected = array![
            [
                QLangComplex::new(sqrt2_inv, 0.0),
                QLangComplex::new(sqrt2_inv, 0.0)
            ],
            [
                QLangComplex::new(sqrt2_inv, 0.0),
                QLangComplex::new(-sqrt2_inv, 0.0)
            ],
        ];

        assert_eq!(h.matrix, expected);
    }

    #[test]
    fn test_hadamard_name() {
        let h = Hadamard::new();
        assert_eq!(h.name(), "Hadamard");
    }


}
