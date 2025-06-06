pub mod batch;
pub mod gates;
pub mod qvm;
pub mod qlang;
pub mod state;
pub mod types;

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
pub extern "C" fn measure(indices: *const usize, len: usize) -> *const usize {
    use std::slice;

    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();
        let input = unsafe { slice::from_raw_parts(indices, len) };

        let joined = input.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
        let line = format!("measure({})", joined);

        qlang.append_from_str(&line);

        match qlang.run_parsed_commands() {
            Ok(Some(result)) => {
                let flat: Vec<usize> = input
                    .iter()
                    .cloned()
                    .zip(result.into_iter())
                    .flat_map(|(q, v)| [q, v as usize])
                    .collect();

                let boxed = flat.into_boxed_slice();
                let ptr = boxed.as_ptr();
                std::mem::forget(boxed);
                return ptr;
            }
            _ => return std::ptr::null(),
        }
    }

    std::ptr::null()
}



#[no_mangle]
pub extern "C" fn measure_all() -> *const u8 {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();

        qlang.append_from_str("measure_all()");

        match qlang.run_parsed_commands() {
            Ok(Some(result)) => {
                let boxed = result.into_boxed_slice();
                let ptr = boxed.as_ptr();
                std::mem::forget(boxed);
                return ptr;
            }
            _ => return std::ptr::null(),
        }
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
    let qlang = QLANG_INSTANCE.get().unwrap().lock().unwrap();
    let source = qlang.ast.iter()
        .map(|cmd| cmd.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    CString::new(source).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn append_qlang_line(line: *const c_char) {
    if let Some(mutex) = QLANG_INSTANCE.get() {
        let mut qlang = mutex.lock().unwrap();
        let c_str = unsafe { CStr::from_ptr(line) };
        let line_str = c_str.to_str().unwrap();

        qlang.append_from_str(line_str);
    }
}
