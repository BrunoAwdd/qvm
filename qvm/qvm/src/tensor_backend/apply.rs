use super::TensorBackend;
use ndarray::Array2;
use qlang_core::{
    gates::quantum_gate_abstract::QuantumGateAbstract, types::qlang_complex::QLangComplex,
};

impl TensorBackend {
    pub fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        self.apply_dense(&gate.matrix(), &[qubit]);
    }

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        self.apply_dense(&gate.matrix(), &[q0, q1]);
    }

    pub fn apply_gate_3q(
        &mut self,
        gate: &dyn QuantumGateAbstract,
        q0: usize,
        q1: usize,
        q2: usize,
    ) {
        self.apply_dense(&gate.matrix(), &[q0, q1, q2]);
    }

    fn apply_dense(&mut self, matrix: &Array2<QLangComplex>, qubits: &[usize]) {
        assert!(
            (1..=3).contains(&qubits.len()),
            "only 1Q, 2Q and 3Q gates are supported"
        );
        for (position, qubit) in qubits.iter().enumerate() {
            assert!(*qubit < self.num_qubits, "qubit index out of range");
            assert!(
                !qubits[..position].contains(qubit),
                "gate qubits must be distinct"
            );
        }

        let state = self.network.to_state_vector();
        let mut next = vec![QLangComplex::zero(); state.len()];
        let local_dimension = 1 << qubits.len();
        assert_eq!(matrix.dim(), (local_dimension, local_dimension));

        for (basis, amplitude) in state.iter().copied().enumerate() {
            let input = qubits
                .iter()
                .fold(0, |index, qubit| (index << 1) | ((basis >> qubit) & 1));
            for output in 0..local_dimension {
                let mut target = basis;
                for (position, qubit) in qubits.iter().enumerate() {
                    let bit = (output >> (qubits.len() - position - 1)) & 1;
                    target = (target & !(1 << qubit)) | (bit << qubit);
                }
                next[target] += matrix[[output, input]] * amplitude;
            }
        }

        self.network.overwrite_from_state_vector(next);
    }
}
