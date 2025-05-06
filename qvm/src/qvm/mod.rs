
pub mod backend;
pub mod cpu_backend;
pub mod cuda_backend;
pub mod tensor_backend;
pub mod cuda;
pub mod util;


mod selected_backend {
    // Se 'cuda' estiver ativado (e for exclusivo), use CUDA
    #[cfg(all(feature = "cuda", not(any(feature = "tensor"))))]
    pub use crate::qvm::cuda_backend::CudaBackend as Backend;

    // Se 'tensor' estiver ativado (e não estiver 'cuda'), use Tensor
    #[cfg(all(feature = "tensor", not(feature = "cuda")))]
    pub use crate::qvm::tensor_backend::TensorBackend as Backend;

    // Se nenhuma feature estiver ativa, use CPU como fallback
    #[cfg(not(any(feature = "cuda", feature = "tensor")))]
    pub use crate::qvm::cpu_backend::CpuBackend as Backend;
}


use selected_backend::Backend;

// traits
use crate::gates::quantum_gate_abstract::QuantumGateAbstract; 
use crate::qvm::backend::QuantumBackend;
use crate::types::qlang_complex::QLangComplex;

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

    pub fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        self.backend.measure_many(qubits) // Mede vários qubits no backend
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
