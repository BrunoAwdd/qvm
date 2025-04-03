pub mod qvm;
pub mod qlang;
pub mod gates;
pub mod state;
pub mod batch;

use crate::qvm::QVM;
use crate::qlang::QLang;
use libc::c_char;
use std::sync::Mutex;
use std::ffi::{CStr, CString};
use once_cell::sync::OnceCell;
use crate::qvm::backend::QuantumBackend;

static QLANG_INSTANCE: OnceCell<Mutex<QLang>> = OnceCell::new();


#[no_mangle]
pub extern "C" fn create_qvm(num_qubits: usize) {
    let qlang = QLang::new(num_qubits);
    let _ = QLANG_INSTANCE.set(Mutex::new(qlang));
}

#[no_mangle]
pub extern "C" fn run_qlang_inline(code: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(code) };
    if let Ok(code_str) = c_str.to_str() {
        if let Some(mutex) = QLANG_INSTANCE.get() {
            let mut qlang = mutex.lock().unwrap();
            qlang.run_from_str(code_str);
        }
    }
}

#[no_mangle]
pub extern "C" fn run_qlang() {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();
        qlang.run();
    }
}

#[no_mangle]
pub extern "C" fn reset_qvm() {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();
        qlang.reset();
    }
}

#[no_mangle]
pub extern "C" fn free_qvm(qvm_ptr: *mut QVM) {
    if !qvm_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(qvm_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn display_qvm() -> *mut c_char {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let qlang = mutex.lock().unwrap();
        let state = qlang.qvm.backend.state_vector();

        let num_qubits = qlang.qvm.num_qubits();
        let mut output = format!("Quantum State with {} qubits:\n", num_qubits);

        for (i, amp) in state.iter().enumerate() {
            output += &format!("|{:0width$b}⟩: {:.4} + {:.4}i\n", i, amp.re, amp.im, width = num_qubits);
        }

        let c_str = CString::new(output).unwrap();
        return c_str.into_raw();
    }
    std::ptr::null_mut()
}


#[no_mangle]
pub extern "C" fn measure_all() -> *const u8 {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();
        let result = qlang.qvm.backend.measure_all();
        return Box::into_raw(result.into_boxed_slice()) as *const u8;
    }
    
    std::ptr::null()
}

#[no_mangle]
pub extern "C" fn free_measurement(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[no_mangle]
pub extern "C" fn free_c_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn get_num_qubits() -> usize {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let qlang = mutex.lock().unwrap();
        return qlang.qvm.num_qubits();
    }
    0
}

#[no_mangle]
pub extern "C" fn get_qlang_source() -> *const c_char {
    let source = QLANG_INSTANCE
    .get()
    .expect("QLANG_INSTANCE não inicializado")
    .lock()
    .unwrap()
    .to_string();
    CString::new(source).unwrap().into_raw()
}