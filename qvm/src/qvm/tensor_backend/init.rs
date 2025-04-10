use crate::state::TensorNetwork;
use super::TensorBackend;

impl TensorBackend {
    pub fn new(num_qubits: usize) -> Self {
        let network = TensorNetwork::new(num_qubits);
        println!("TensorBackend: network created with {} qubits", num_qubits);
        Self { network, num_qubits }
    }
}
