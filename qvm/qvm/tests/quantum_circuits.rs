use qlang_core::gates::{
    one_q::{hadamard::Hadamard, pauli_x::PauliX},
    three_q::{fredkin::Fredkin, toffoli::Toffoli},
    two_q::cnot::CNOT,
};
use qvm::QVM;

const EPSILON: f64 = 1e-12;

fn assert_amplitude(qvm: &QVM, basis: usize, re: f64) {
    let state = qvm.state_vector();
    for (index, amplitude) in state.iter().enumerate() {
        let expected = if index == basis { re } else { 0.0 };
        assert!(
            (amplitude.re - expected).abs() < EPSILON && amplitude.im.abs() < EPSILON,
            "unexpected amplitude at |{index}>: {amplitude:?}"
        );
    }
}

#[test]
fn bell_state_has_expected_amplitudes() {
    let mut qvm = QVM::new(2);
    qvm.apply_gate(&Hadamard::new(), 0);
    qvm.apply_gate_2q(&CNOT::new(), 0, 1);

    let state = qvm.state_vector();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state[0].re - expected).abs() < EPSILON);
    assert!((state[3].re - expected).abs() < EPSILON);
    assert!(state[1].norm_sqr() < EPSILON);
    assert!(state[2].norm_sqr() < EPSILON);
}

#[test]
fn ghz_state_has_expected_amplitudes() {
    let mut qvm = QVM::new(3);
    qvm.apply_gate(&Hadamard::new(), 0);
    qvm.apply_gate_2q(&CNOT::new(), 0, 1);
    qvm.apply_gate_2q(&CNOT::new(), 1, 2);

    let state = qvm.state_vector();
    let expected = 1.0 / 2.0_f64.sqrt();
    for (index, amplitude) in state.iter().enumerate() {
        let expected_at_index = if index == 0 || index == 7 {
            expected
        } else {
            0.0
        };
        assert!((amplitude.re - expected_at_index).abs() < EPSILON);
        assert!(amplitude.im.abs() < EPSILON);
    }
}

#[test]
fn toffoli_transforms_the_state_instead_of_adding_to_it() {
    let mut qvm = QVM::new(3);
    qvm.apply_gate(&PauliX::new(), 0);
    qvm.apply_gate(&PauliX::new(), 1);
    qvm.apply_gate_3q(&Toffoli::new(), 0, 1, 2);

    assert_amplitude(&qvm, 7, 1.0);
}

#[test]
fn fredkin_transforms_the_state_instead_of_adding_to_it() {
    let mut qvm = QVM::new(3);
    qvm.apply_gate(&PauliX::new(), 0);
    qvm.apply_gate(&PauliX::new(), 1);
    qvm.apply_gate_3q(&Fredkin::new(), 0, 1, 2);

    assert_amplitude(&qvm, 5, 1.0);
}

#[test]
fn cloning_preserves_the_quantum_state() {
    let mut qvm = QVM::new(2);
    qvm.apply_gate(&Hadamard::new(), 0);
    qvm.apply_gate_2q(&CNOT::new(), 0, 1);

    let cloned = qvm.clone();
    assert_eq!(qvm.state_vector(), cloned.state_vector());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_box_clone_preserves_the_quantum_state() {
    use qvm::{backend::QuantumBackend, cuda_backend::CudaBackend};

    let mut backend = CudaBackend::new(1);
    backend.apply_gate(&PauliX::new(), 0);
    let cloned = QuantumBackend::box_clone(&backend);

    assert_eq!(backend.state_vector(), cloned.state_vector());
}
