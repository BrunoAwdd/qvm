use ndarray::{array, Array2};
use std::f64::consts::PI;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// The T gate — also known as the π/8 gate.
///
/// This gate applies a phase shift of π/4 to the `|1⟩` state. It is represented by:
///
/// ```text
/// | 1         0 |
/// | 0  e^(iπ/4) |
/// ```
///
/// It is a fundamental gate for universal quantum computing, and is the square root of the S gate:
/// - `T * T = S`
/// - `T†` is its inverse (−π/4 phase).
pub struct T {
    /// The 2x2 matrix representation of the T gate.
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for T {
    /// Returns the matrix of the T gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the gate name.
    fn name(&self) -> &'static str {
        "T"
    }
}

impl T {
    /// Constructs a new T gate.
    pub fn new() -> Self {
        let angle = PI / 4.0;
        let phase = QLangComplex::new(angle.cos(), angle.sin()); // e^(iπ/4)

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let matrix = array![
            [one, zero],
            [zero, phase]
        ];

        Self { matrix }
    }
}
