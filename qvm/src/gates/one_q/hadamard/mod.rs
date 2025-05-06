use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

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
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the name of the gate.
    fn name(&self) -> &'static str {
        "Hadamard"
    }
}

impl Hadamard {
    /// Constructs a new Hadamard gate.
    pub fn new() -> Self {
        let factor: f64 = 1.0 / (2.0_f64).sqrt();
        let matrix = array![
            [QLangComplex::new(factor, 0.0), QLangComplex::new(factor, 0.0)],
            [QLangComplex::new(factor, 0.0), QLangComplex::new(-factor, 0.0)]
        ];

        Self { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::one_q::hadamard::Hadamard;
    use crate::types::qlang_complex::QLangComplex;
    use crate::qvm::QVM;
    use crate::qlang::QLang;
    use ndarray::array;

    #[test]
    fn test_hadamard_matrix() {
        let h = Hadamard::new();
        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();

        let expected = array![
            [QLangComplex::new(sqrt2_inv, 0.0), QLangComplex::new(sqrt2_inv, 0.0)],
            [QLangComplex::new(sqrt2_inv, 0.0), QLangComplex::new(-sqrt2_inv, 0.0)],
        ];

        assert_eq!(h.matrix, expected);
    }

    #[test]
    fn test_hadamard_name() {
        let h = Hadamard::new();
        assert_eq!(h.name(), "Hadamard");
    }

    #[test]
    fn test_hadamard_apply() {
        let mut qvm = QVM::new(1);
        qvm.apply_gate(&Hadamard::new(), 0);

        let mut count = 0;
        for _ in 0..10 {
            let mut copy = qvm.clone();
            count += copy.measure(0) as usize;
        }

        println!("Probability of 1: {}", count as f64 / 10.0);
    }

    #[test]
    fn test_hadamard_distribution() {
        let mut count_0 = 0;
        let mut count_1 = 0;
        let iterations = 100;

        for _ in 0..iterations {
            let mut qlang = QLang::new(1);
            qlang.append_from_str("h(0)");
            qlang.append_from_str( "run()");

            qlang.run_parsed_commands().unwrap();

            let result = qlang.qvm.measure(0);

            if result == 0 {
                count_0 += 1;
            } else {
                count_1 += 1;
            }
        }

        let prob_0 = count_0 as f64 / iterations as f64;
        let prob_1 = count_1 as f64 / iterations as f64;

        // Assert that both probabilities are roughly close to 0.5 (±10%)
        assert!((0.4..=0.6).contains(&prob_0), "Probability of 0 out of range");
        assert!((0.4..=0.6).contains(&prob_1), "Probability of 1 out of range");
    }

    #[test]
    fn test_measure_many_hadamard() {
        let mut qlang = QLang::new(3);

        qlang.append_from_str("measure(0, 1, 2)");
        let _m0 = qlang.run_parsed_commands(); 

        qlang.append_from_str("h(0)");
        qlang.append_from_str("measure(0, 1, 2)");
        let _m1 = qlang.run_parsed_commands(); 
        qlang.append_from_str("x(1)");
        qlang.append_from_str("measure(0, 1, 2)");
        let _m2 = qlang.run_parsed_commands(); 

        qlang.append_from_str("measure_all()");

        let _m_all = qlang.run_parsed_commands(); 

        qlang.run(); // Executa AST até esse ponto

        let result = qlang.qvm.measure_all();

        assert!(result[0] == 0 || result[0] == 1, "Invalid Result");
    }

}
