
pub mod backend;
pub mod cpu_backend;
pub mod cuda_backend;
pub mod cuda;
pub mod util;

#[cfg_attr(feature = "cuda", path = "cuda_backend.rs")]
#[cfg_attr(feature = "wgpu", path = "wgpu_backend.rs")]
#[cfg_attr(feature = "cpu", path = "cpu_backend.rs")]
mod selected_backend;

use selected_backend::Backend;

// traits
use crate::gates::quantum_gate_abstract::QuantumGateAbstract; 
use crate::qvm::backend::QuantumBackend;
use crate::types::qlang_complex::QLangComplex;

#[cfg(feature = "cuda")]
use crate::qvm::cuda_backend::CudaBackend;


pub struct QVM {
    pub backend: Backend, // Backend do QVM
}

impl QVM {
    /// Inicializa o QVM com um dado número de qubits
    pub fn new(num_qubits: usize) -> Self {
        let backend: Backend = Backend::new(num_qubits);
        Self { backend }
    }

    /// Aplica uma porta quântica ao QVM
    pub fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        self.backend.apply_gate( gate, qubit);
    }

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        self.backend.apply_gate_2q( gate, q0, q1); // Aplica o gate ao backend
    }

    pub fn apply_gate_3q(&mut self, gate: &impl QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        self.backend.apply_gate_3q(gate, q0, q1, q2);
    }

    pub fn measure(&mut self, qubit: usize) -> u8 {
        self.backend.measure(qubit) // Mede um qubit no backend
    }

    /// Mede o estado de todos os qubits
    pub fn measure_all(&mut self) -> Vec<u8> {
        self.backend.measure_all() // Mede todos os qubits no backend
    }

    /// Exibe o estado do QVM
    pub fn display(&self) {
        self.backend.display(); // Mostra o estado atual do backend
    }

    pub fn num_qubits(&self) -> usize {
        self.backend.num_qubits()
    }

    pub fn state_vector(&self) -> Vec<QLangComplex> {
        self.backend.state_vector()
    }


}

impl Clone for QVM {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
        }
    }
}
