use qlang::qvm::QVM;
use qlang::gates::one_q::pauli_x::PauliX;

#[test]
fn test_qvm_basic() {
    let mut qvm = QVM::new(1);
    qvm.apply_gate(&PauliX::new(), 0);
    let result = qvm.measure(0);
    assert_eq!(result, 1);
}

#[test]
fn test_qvm_clone() {
    let mut qvm = QVM::new(1);
    qvm.apply_gate(&PauliX::new(), 0);

    let clone = qvm.clone();
    assert_eq!(clone.num_qubits(), 1);
}
