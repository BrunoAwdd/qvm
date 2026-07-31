use std::{io::Write, process::Command};

use tempfile::NamedTempFile;

fn source_file(source: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    file
}

#[test]
fn check_accepts_a_valid_program() {
    let file = source_file("create(2)\nh(0)\ncnot(0, 1)\n");
    let output = Command::new(env!("CARGO_BIN_EXE_qlang"))
        .arg("check")
        .arg(file.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("OK:"));
}

#[test]
fn check_rejects_invalid_syntax() {
    let file = source_file("create(2)\nh(\n");
    let output = Command::new(env!("CARGO_BIN_EXE_qlang"))
        .arg("check")
        .arg(file.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
}

#[test]
fn run_executes_a_valid_program() {
    let file = source_file("create(1)\nx(0)\n");
    let status = Command::new(env!("CARGO_BIN_EXE_qlang"))
        .arg("run")
        .arg(file.path())
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn check_accepts_typed_functions() {
    let file = source_file(
        "fn rotate(q: qubit, angle: float) -> void {\n  rx(q, angle)\n}\nrotate(0, 0.5)\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_qlang"))
        .arg("check")
        .arg(file.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn check_rejects_semantic_type_errors() {
    let file = source_file("let value: int = 1\nvalue = 1.5\n");
    let output = Command::new(env!("CARGO_BIN_EXE_qlang"))
        .arg("check")
        .arg(file.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot assign"));
}
