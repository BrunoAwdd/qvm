
pub mod backend;
pub mod cuda_backend;
pub mod cpu_backend;
pub mod cuda;
pub mod util;

use crate::gates::quantum_gate_abstract::QuantumGateAbstract; // Traço que define os gates

#[cfg_attr(feature = "cuda", path = "cuda_backend.rs")]
#[cfg_attr(feature = "wgpu", path = "wgpu_backend.rs")]
#[cfg_attr(feature = "cpu", path = "cpu_backend.rs")]
mod selected_backend;

use selected_backend::Backend;

use crate::qvm::backend::QuantumBackend;

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

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize) {
        self.backend.apply_gate_2q( gate, q1, q2); // Aplica o gate ao backend
    }

    pub fn apply_gate_3q(&mut self, gate: &impl QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        self.backend.apply_gate_3q(gate, q0, q1, q2);
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


}
