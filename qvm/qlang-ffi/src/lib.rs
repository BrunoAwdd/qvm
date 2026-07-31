// The exported C ABI validates every pointer before dereferencing it. Marking
// these functions `unsafe` would not add protection for C/Python callers and
// would make the Rust declaration diverge from the public header.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::{Mutex, OnceLock},
};

use qlang::QLang;
use qvm::backend::QuantumBackend;
use serde_json::json;

static INSTANCE: OnceLock<Mutex<Option<QLang>>> = OnceLock::new();

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn instance() -> &'static Mutex<Option<QLang>> { INSTANCE.get_or_init(|| Mutex::new(None)) }

fn set_error(message: impl Into<String>) {
    let message = message.into().replace('\0', "\\0");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(message).ok());
}

fn status(operation: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
            0
        }
        Ok(Err(error)) => {
            set_error(error);
            1
        }
        Err(_) => {
            set_error("QLang panicked while handling an FFI call");
            2
        }
    }
}

unsafe fn source_from_ptr<'a>(source: *const c_char) -> Result<&'a str, String> {
    if source.is_null() {
        return Err("source pointer is null".into());
    }
    CStr::from_ptr(source)
        .to_str()
        .map_err(|error| format!("source is not valid UTF-8: {error}"))
}

#[no_mangle]
pub extern "C" fn qlang_create(num_qubits: usize) -> i32 {
    status(|| {
        if num_qubits >= usize::BITS as usize {
            return Err(format!("num_qubits must be less than {}", usize::BITS));
        }
        *instance().lock().map_err(|_| "QLang mutex is poisoned")? = Some(QLang::new(num_qubits));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn qlang_run_source(source: *const c_char) -> i32 {
    status(|| {
        let source = unsafe { source_from_ptr(source)? };
        let mut guard = instance().lock().map_err(|_| "QLang mutex is poisoned")?;
        let qlang = guard
            .as_mut()
            .ok_or("QLang is not initialized; call qlang_create first")?;
        qlang.clear_program();
        qlang.append_from_lines(source.lines());
        qlang.run_parsed_commands()?;
        qlang.try_run()
    })
}

#[no_mangle]
pub extern "C" fn qlang_reset() -> i32 {
    status(|| {
        let mut guard = instance().lock().map_err(|_| "QLang mutex is poisoned")?;
        let qlang = guard.as_mut().ok_or("QLang is not initialized")?;
        qlang.reset();
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn qlang_num_qubits() -> usize {
    instance()
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|qlang| qlang.qvm.num_qubits()))
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn qlang_measure_all(output: *mut u8, capacity: usize) -> isize {
    let mut written = -1;
    let result = status(|| {
        let mut guard = instance().lock().map_err(|_| "QLang mutex is poisoned")?;
        let qlang = guard.as_mut().ok_or("QLang is not initialized")?;
        let required = qlang.qvm.num_qubits();
        if output.is_null() {
            return Err("measurement output pointer is null".into());
        }
        if capacity < required {
            return Err(format!(
                "measurement buffer needs {required} bytes, got {capacity}"
            ));
        }
        let measurements = qlang.qvm.measure_all();
        unsafe { ptr::copy_nonoverlapping(measurements.as_ptr(), output, measurements.len()) };
        written = measurements.len() as isize;
        Ok(())
    });
    if result == 0 {
        written
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn qlang_state_json() -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<CString, String> {
        let guard = instance().lock().map_err(|_| "QLang mutex is poisoned")?;
        let qlang = guard.as_ref().ok_or("QLang is not initialized")?;
        let amplitudes: Vec<_> = qlang
            .qvm
            .state_vector()
            .into_iter()
            .map(|amplitude| json!({"re": amplitude.re, "im": amplitude.im}))
            .collect();
        CString::new(
            json!({
                "num_qubits": qlang.qvm.num_qubits(),
                "backend": qlang.qvm.backend.name(),
                "amplitudes": amplitudes,
            })
            .to_string(),
        )
        .map_err(|error| error.to_string())
    }));
    match result {
        Ok(Ok(value)) => value.into_raw(),
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("QLang panicked while serializing state");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn qlang_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(ptr::null(), |message| message.as_ptr())
    })
}

#[no_mangle]
pub extern "C" fn qlang_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe { drop(CString::from_raw(value)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ffi_executes_and_serializes_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert_eq!(qlang_create(1), 0);
        let source = CString::new("x(0)").unwrap();
        assert_eq!(qlang_run_source(source.as_ptr()), 0);
        let json = qlang_state_json();
        assert!(!json.is_null());
        let value: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        assert_eq!(value["num_qubits"], 1);
        assert_eq!(value["amplitudes"][1]["re"], 1.0);
        qlang_string_free(json);
    }

    #[test]
    fn ffi_does_not_replay_previous_source() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert_eq!(qlang_create(1), 0);
        let source = CString::new("x(0)").unwrap();
        assert_eq!(qlang_run_source(source.as_ptr()), 0);
        assert_eq!(qlang_run_source(source.as_ptr()), 0);

        let json = qlang_state_json();
        let value: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        assert_eq!(value["amplitudes"][0]["re"], 1.0);
        assert_eq!(value["amplitudes"][1]["re"], 0.0);
        qlang_string_free(json);
    }

    #[test]
    fn ffi_reports_invalid_source() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert_eq!(qlang_create(1), 0);
        let source = CString::new("h(").unwrap();
        assert_ne!(qlang_run_source(source.as_ptr()), 0);
        assert!(!qlang_last_error().is_null());
    }
}
