use qvm::state::quantum_state::QuantumState;
use qvm::gates::hadamard::Hadamard;

#[test]
fn test_measure_all_qubits() {
    let mut state = QuantumState::new(2); // Dois qubits

    let h_gate = Hadamard::new();
    state.apply_gate(&h_gate, 0); // Aplica Hadamard no primeiro qubit
    state.apply_gate(&h_gate, 1); // Aplica Hadamard no segundo qubit

    let results = state.measure_all();

    assert!(results.len() == 2);
    assert!(results[0] == 0 || results[0] == 1);
    assert!(results[1] == 0 || results[1] == 1);
}
