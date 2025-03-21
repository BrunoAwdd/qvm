use qvm::state::quantum_state::QuantumState;
use qvm::gates::hadamard::Hadamard;

#[test]
fn test_measure_qubit() {
    let mut state: QuantumState = QuantumState::new(1); // Um único qubit

    let h_gate: Hadamard = Hadamard::new();
    state.apply_gate(&h_gate, 0); // Aplica Hadamard no qubit 0

    let result: u8 = state.measure(0);

    assert!(result == 0 || result == 1); // Deve ser 0 ou 1
}
