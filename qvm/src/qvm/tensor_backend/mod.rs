// src/qvm/tensor_backend/mod.rs

pub mod apply;
pub mod fallback;
pub mod init;
pub mod measure;
pub mod contract;

use crate::{
    state::tensor_network::TensorNetwork,
    qvm::backend::QuantumBackend,
    gates::quantum_gate_abstract::QuantumGateAbstract,
    types::qlang_complex::QLangComplex,
};

pub struct TensorBackend {
    pub network: TensorNetwork,
    pub num_qubits: usize,
}

impl QuantumBackend for TensorBackend {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn state_vector(&self) -> Vec<QLangComplex> {
        self.network.to_state_vector()
    }

    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        self.apply_gate(gate, qubit);
    }

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        self.apply_gate_2q(gate, q0, q1);
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        self.apply_gate_3q(gate, q0, q1, q2);
    }

    fn measure(&mut self, qubit: usize) -> u8 {
        self.measure(qubit)
    }

    fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        self.measure_many(qubits)
    }

    fn measure_all(&mut self) -> Vec<u8> {
        self.measure_all()
    }

    fn display(&self) {
        let state = self.state_vector();
        let n = self.num_qubits;

        println!("🔬 Estado quântico ({} qubits):", n);
        for (i, amp) in state.iter().enumerate() {
            if amp.norm_sqr() > 1e-10 {
                let bits = format!("{:0width$b}", i, width = n);
                println!("|{}⟩: {:.4} + {:.4}i", bits, amp.re, amp.im);
            }
        }
    }

    fn reset(&mut self, num_qubits: usize) {
        *self = TensorBackend::new(num_qubits);
    }

    fn box_clone(&self) -> Box<dyn QuantumBackend> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "Tensor"
    }
}

impl Clone for TensorBackend {
    fn clone(&self) -> Self {
        // Clonagem ingênua por enquanto
        TensorBackend::new(self.num_qubits)
    }
}

pub type Backend = TensorBackend;
