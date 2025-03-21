use crate::state::quantum_state::QuantumState;
use crate::gates::quantum_gate_abstract::{QuantumGateAbstract}; // Traço que define os gates

pub struct QVM {
    pub state: QuantumState, // Estado quântico
}

impl QVM {
    /// Inicializa o QVM com um dado número de qubits
    pub fn new(num_qubits: usize) -> Self {
        let state: QuantumState = QuantumState::new(num_qubits);
        Self { state }
    }

    /// Aplica uma porta quântica ao QVM
    pub fn apply_gate<T: QuantumGateAbstract>(&mut self, gate: &T, qubit: usize) {
        self.state.apply_gate(gate, qubit); // Aplica o gate ao estado quântico
    }

    /// Mede o estado de todos os qubits
    pub fn measure_all(&mut self) -> Vec<u8> {
        self.state.measure_all() // Mede todos os qubits no estado
    }

    /// Exibe o estado do QVM
    pub fn display(&self) {
        self.state.display(); // Mostra o estado atual dos qubits
    }
}
