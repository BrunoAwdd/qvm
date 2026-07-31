// The exported C ABI validates every pointer before dereferencing it. Marking
// these functions `unsafe` would not add protection for C/Python callers and
// would make the Rust declaration diverge from the public header.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::Mutex,
};

use qlang::QLang;
use qvm::backend::QuantumBackend;
use serde_json::json;

#[repr(C)]
pub struct QLangHandle {
    runtime: Mutex<QLang>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

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

unsafe fn handle_from_ptr<'a>(handle: *mut QLangHandle) -> Result<&'a QLangHandle, String> {
    handle
        .as_ref()
        .ok_or_else(|| "QLang handle is null or has been closed".into())
}

#[no_mangle]
pub extern "C" fn qlang_create(num_qubits: usize) -> *mut QLangHandle {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<QLangHandle, String> {
        if num_qubits >= usize::BITS as usize {
            return Err(format!("num_qubits must be less than {}", usize::BITS));
        }
        Ok(QLangHandle {
            runtime: Mutex::new(QLang::new(num_qubits)),
        })
    }));
    match result {
        Ok(Ok(handle)) => {
            LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
            Box::into_raw(Box::new(handle))
        }
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("QLang panicked while creating an FFI handle");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn qlang_destroy(handle: *mut QLangHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[no_mangle]
pub extern "C" fn qlang_run_source(handle: *mut QLangHandle, source: *const c_char) -> i32 {
    status(|| {
        let source = unsafe { source_from_ptr(source)? };
        let handle = unsafe { handle_from_ptr(handle)? };
        let mut qlang = handle
            .runtime
            .lock()
            .map_err(|_| "QLang handle mutex is poisoned")?;
        qlang.clear_program();
        qlang.append_from_lines(source.lines());
        qlang.run_parsed_commands()?;
        qlang.try_run()
    })
}

#[no_mangle]
pub extern "C" fn qlang_reset(handle: *mut QLangHandle) -> i32 {
    status(|| {
        let handle = unsafe { handle_from_ptr(handle)? };
        let mut qlang = handle
            .runtime
            .lock()
            .map_err(|_| "QLang handle mutex is poisoned")?;
        qlang.reset();
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn qlang_num_qubits(handle: *mut QLangHandle) -> usize {
    unsafe { handle_from_ptr(handle) }
        .ok()
        .and_then(|handle| handle.runtime.lock().ok())
        .map(|qlang| qlang.qvm.num_qubits())
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn qlang_measure_all(
    handle: *mut QLangHandle,
    output: *mut u8,
    capacity: usize,
) -> isize {
    let mut written = -1;
    let result = status(|| {
        let handle = unsafe { handle_from_ptr(handle)? };
        let mut qlang = handle
            .runtime
            .lock()
            .map_err(|_| "QLang handle mutex is poisoned")?;
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
pub extern "C" fn qlang_state_json(handle: *mut QLangHandle) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<CString, String> {
        let handle = unsafe { handle_from_ptr(handle)? };
        let qlang = handle
            .runtime
            .lock()
            .map_err(|_| "QLang handle mutex is poisoned")?;
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

    #[test]
    fn ffi_executes_and_serializes_state() {
        let handle = qlang_create(1);
        assert!(!handle.is_null());
        let source = CString::new("x(0)").unwrap();
        assert_eq!(qlang_run_source(handle, source.as_ptr()), 0);
        let json = qlang_state_json(handle);
        assert!(!json.is_null());
        let value: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        assert_eq!(value["num_qubits"], 1);
        assert_eq!(value["amplitudes"][1]["re"], 1.0);
        qlang_string_free(json);
        qlang_destroy(handle);
    }

    #[test]
    fn ffi_does_not_replay_previous_source() {
        let handle = qlang_create(1);
        assert!(!handle.is_null());
        let source = CString::new("x(0)").unwrap();
        assert_eq!(qlang_run_source(handle, source.as_ptr()), 0);
        assert_eq!(qlang_run_source(handle, source.as_ptr()), 0);

        let json = qlang_state_json(handle);
        let value: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        assert_eq!(value["amplitudes"][0]["re"], 1.0);
        assert_eq!(value["amplitudes"][1]["re"], 0.0);
        qlang_string_free(json);
        qlang_destroy(handle);
    }

    #[test]
    fn ffi_reports_invalid_source() {
        let handle = qlang_create(1);
        assert!(!handle.is_null());
        let source = CString::new("h(").unwrap();
        assert_ne!(qlang_run_source(handle, source.as_ptr()), 0);
        assert!(!qlang_last_error().is_null());
        qlang_destroy(handle);
    }

    #[test]
    fn ffi_handles_keep_independent_states() {
        let first = qlang_create(1);
        let second = qlang_create(2);
        assert!(!first.is_null());
        assert!(!second.is_null());
        let source = CString::new("x(0)").unwrap();
        assert_eq!(qlang_run_source(first, source.as_ptr()), 0);

        assert_eq!(qlang_num_qubits(first), 1);
        assert_eq!(qlang_num_qubits(second), 2);
        let first_json = qlang_state_json(first);
        let second_json = qlang_state_json(second);
        let first_state: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(first_json) }.to_str().unwrap()).unwrap();
        let second_state: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(second_json) }.to_str().unwrap()).unwrap();
        assert_eq!(first_state["amplitudes"][1]["re"], 1.0);
        assert_eq!(second_state["amplitudes"][0]["re"], 1.0);
        qlang_string_free(first_json);
        qlang_string_free(second_json);
        qlang_destroy(first);
        qlang_destroy(second);
    }
}
