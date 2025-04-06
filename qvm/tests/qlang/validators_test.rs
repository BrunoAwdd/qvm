use qlang::qlang::validators::validate_gate_arity;

#[test]
fn test_hadamard_valid() {
    let args = vec!["0"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("hadamard", 1, &args_refs).is_ok());
}

#[test]
fn test_u1_valid() {
    let args = vec!["0", "3.14"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("u1", 1, &args_refs).is_ok());
}
#[test]
fn test_u1_invalid_qubit() {
    let args = vec!["abc", "3.14"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    let result = validate_gate_arity("u1", 3, &args_refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid qubit index"));
}


#[test]
fn test_u2_invalid() {
    let args = vec!["0", "3.14"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("u2", 1, &args_refs).is_err());
}

#[test]
fn test_u3_valid() {
    let args = vec!["0", "3.14", "1.57", "0.0"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("u3", 1, &args_refs).is_ok());
}

#[test]
fn test_rx_valid() {
    let args = vec!["0", "3.14"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("rx", 1, &args_refs).is_ok());
}
#[test]
fn test_rx_invalid_float() {
    let args = vec!["0", "not_a_number"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    let result = validate_gate_arity("rx", 1, &args_refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid theta"));
}


#[test]
fn test_cnot_valid() {
    let args = vec!["0", "1"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("cnot", 2, &args_refs).is_ok());
}

#[test]
fn test_cnot_out_of_bounds() {
    let args = vec!["0", "3"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    let result = validate_gate_arity("cnot", 2, &args_refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of bounds"));
}

#[test]
fn test_cnot_same_qubit() {
    let args = vec!["1", "1"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    let result = validate_gate_arity("cnot", 2, &args_refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("must be different"));
}

#[test]
fn test_swap_wrong_arity() {
    let args = vec!["0"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    assert!(validate_gate_arity("swap", 2, &args_refs).is_err());
}


#[test]
fn test_toffoli_valid() {
    let args = vec!["0", "1", "2"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect::<Vec<&str>>();
    assert!(validate_gate_arity("toffoli", 4, &args_refs).is_ok());
}

#[test]
fn test_toffoli_same_qubits() {
    let args = vec!["1", "1", "1"];
    let args_refs: Vec<&str> = args.iter().map(|s| &**s).collect();
    let result = validate_gate_arity("toffoli", 3, &args_refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("must be distinct"));
}
