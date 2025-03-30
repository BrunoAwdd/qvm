//#[cfg(test)]
//mod tests;

pub mod qvm;
pub mod qlang;
pub mod gates;
pub mod state;

use crate::qvm::QVM;
use crate::gates::{hadamard::Hadamard, pauli_x::PauliX, pauli_y::PauliY, pauli_z::PauliZ, cnot::CNOT};

#[no_mangle]
pub extern "C" fn create_qvm(num_qubits: usize) -> *mut QVM {
    let qvm = QVM::new(num_qubits);
    Box::into_raw(Box::new(qvm))
}

#[no_mangle]
pub extern "C" fn apply_hadamard(qvm_ptr: *mut QVM, qubit: usize) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let h_gate = Hadamard::new();
        qvm.apply_gate(&h_gate, qubit);
    }
}

#[no_mangle]
pub extern "C" fn apply_pauli_x(qvm_ptr: *mut QVM, qubit: usize) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let p_x_gate = PauliX::new();
        qvm.apply_gate(&p_x_gate, qubit);
    }
}

#[no_mangle]
pub extern "C" fn apply_pauli_y(qvm_ptr: *mut QVM, qubit: usize) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let p_y_gate = PauliY::new();
        qvm.apply_gate(&p_y_gate, qubit);
    }
}

#[no_mangle]
pub extern "C" fn apply_pauli_z(qvm_ptr: *mut QVM, qubit: usize) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let p_z_gate = PauliZ::new();
        qvm.apply_gate(&p_z_gate, qubit);
    }
}

#[no_mangle]
pub extern "C" fn apply_cnot(qvm_ptr: *mut QVM, control_qubit: usize, target_qubit: usize) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let cnot_gate = CNOT::new();
        qvm.apply_gate_2q(&cnot_gate, control_qubit, target_qubit);
    }
}

#[no_mangle]
pub extern "C" fn measure_all(qvm_ptr: *mut QVM) -> *mut u8 {
    if qvm_ptr.is_null() {
        println!("Erro: Ponteiro nulo passado para measure_all");
        return std::ptr::null_mut();
    }

    let qvm = unsafe { &mut *qvm_ptr };
    let result = qvm.measure_all();
    

    // Clonamos o resultado e o transformamos em um Box<[u8]>
    let mut result_box = result.into_boxed_slice();
    let result_ptr = result_box.as_mut_ptr(); // Obtemos o ponteiro mutável para o array

    // Evita que o Rust libere a memória automaticamente
    std::mem::forget(result_box);
    
    result_ptr
}

#[no_mangle]
pub extern "C" fn get_num_qubits(qvm_ptr: *mut QVM) -> usize {
    if qvm_ptr.is_null() { return 0; }
    let qvm = unsafe { &*qvm_ptr };
    qvm.state.num_qubits
}

#[no_mangle]
pub extern "C" fn display_qvm(qvm_ptr: *mut QVM) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        qvm.display();
    }
}

#[no_mangle]
pub extern "C" fn run_qlang_inline(code: *const libc::c_char) {
    use std::ffi::CStr;
    if code.is_null() { return; }
    let c_str = unsafe { CStr::from_ptr(code) };
    if let Ok(code_str) = c_str.to_str() {
        let mut qlang = crate::qlang::QLang::new(2);
        qlang.run_from_str(code_str);
    }
}

#[no_mangle]
pub extern "C" fn free_qvm(qvm_ptr: *mut QVM) {
    if !qvm_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(qvm_ptr)); 
        }
    }
}

#[no_mangle]
pub extern "C" fn reset_qvm(qvm_ptr: *mut QVM) {
    if !qvm_ptr.is_null() {
        let qvm = unsafe { &mut *qvm_ptr };
        let num = qvm.state.num_qubits;
        qvm.state = crate::state::quantum_state::QuantumState::new(num);
    }
}

