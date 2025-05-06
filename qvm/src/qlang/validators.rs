pub fn validate_gate_arity(name: &str, total_qubits: usize, args: &[&str], ) -> Result<(), String>{
    match name {
        "hadamard" | "identity" | "paulix" | "pauliy" | 
        "pauliz" | "s" | "s_dagger" | "t" | "t_dagger" =>  validate_1q_gate(name, total_qubits, args),
        "u1" => validate_1q_gate_1f4(name, total_qubits, args),
        "u2" => validate_1q_gate_2f4(name, total_qubits, args),
        "u3" => validate_1q_gate_3f4(name, total_qubits, args),
        "phase" |"rx" | "ry" | "rz" => validate_1q_gate_1f4(name, total_qubits, args),
        "cnot" | "swap" | "cy" | "cz" | "iswap" => validate_2q_gate(name, total_qubits, args),
        "toffoli" | "fredkin" => validate_3q_gate(name, total_qubits, args),
        "controlled_u" | "cu" => validate_controlled_u(name, total_qubits, args),
        _ => Err(format!("Unknown gate: '{}'", name)),
    }
}

/// Validates a single-qubit gate call with one qubit index (e.g., `x(0)`).
///
/// # Parameters
/// - `name`: Name of the gate (used for error reporting).
/// - `total_qubits`: Total number of available qubits in the QVM.
/// - `args`: Arguments passed to the gate, expected to be `[qubit_index]`.
///
/// # Returns
/// - `Ok(())` if the argument is valid and within bounds.
/// - `Err(String)` if the qubit index is missing, invalid, or out of bounds.
///
/// # Example
/// ```ignore
/// validate_1q_gate("x", 3, &["0"])?;    // OK
/// validate_1q_gate("x", 3, &["5"])?;    // Error: out of bounds
/// validate_1q_gate("x", 3, &[])?;       // Error: missing argument
/// ```
fn validate_1q_gate(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 qubit indices", name));
    }

    let qubit = args[0].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[0]))?;
    check_bounds(name, qubit, total_qubits)?;

    Ok(())
}

/// Validates a single-qubit gate call with one qubit index and one float parameter (e.g., `rx(0, 3.14)`).
///
/// # Parameters
/// - `name`: Gate name (for error messages).
/// - `total_qubits`: Number of available qubits.
/// - `args`: Arguments, expected as `[qubit_index, theta]`.
///
/// # Returns
/// - `Ok(())` if both arguments are valid.
/// - `Err(String)` if the index is invalid/out of bounds, or `theta` is not a valid float.
///
/// # Example
/// ```ignore
/// validate_1q_gate_1f4("rx", 3, &["1", "3.14"])?;  // OK
/// validate_1q_gate_1f4("rx", 3, &["1", "abc"])?;   // Error: invalid theta
/// ```
fn validate_1q_gate_1f4(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 2 {
        return Err(format!("{} requires 2 arguments: qubit, theta", name));
    }

    let qubit = args[0].parse::<usize>()
        .map_err(|_| format!("{}: invalid qubit index '{}'", name, args[0]))?;

    check_bounds(name, qubit, total_qubits)?;

    args[1].parse::<f64>()
        .map_err(|_| format!("{}: invalid theta value '{}'", name, args[1]))?;

    Ok(())
}

/// Validates a single-qubit gate call with one index and two float parameters (e.g., `u2(0, 1.0, 0.5)`).
///
/// # Parameters
/// - `name`: Gate name (e.g., `"u2"`).
/// - `total_qubits`: Total qubits in the system.
/// - `args`: Arguments, expected as `[qubit_index, phi, lambda]`.
///
/// # Returns
/// - `Ok(())` if all arguments are valid.
/// - `Err(String)` if any are missing, malformed, or out of range.
///
/// # Example
/// ```ignore
/// validate_1q_gate_2f4("u2", 4, &["0", "0.5", "0.1"])?; // OK
/// validate_1q_gate_2f4("u2", 4, &["a", "b", "c"])?;     // Error
/// ```
fn validate_1q_gate_2f4(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 3 {
        return Err("u2 requires 3 arguments: qubit, phi, lambda".into());
    }

    let qubit = args[0].parse::<usize>()
        .map_err(|_| format!("u2: invalid qubit index '{}'", args[0]))?;

    check_bounds(name, qubit, total_qubits)?;

    args[1].parse::<f64>()
        .map_err(|_| format!("u2: invalid phi value '{}'", args[1]))?;
    args[2].parse::<f64>()
        .map_err(|_| format!("u2: invalid lambda value '{}'", args[2]))?;

    Ok(())
}

/// Validates a single-qubit gate call with one index and three float parameters (e.g., `u3(0, π, π/2, π/4)`).
///
/// # Parameters
/// - `name`: Gate name (e.g., `"u3"`).
/// - `total_qubits`: Number of available qubits.
/// - `args`: Arguments, expected as `[qubit_index, theta, phi, lambda]`.
///
/// # Returns
/// - `Ok(())` if all inputs are correct.
/// - `Err(String)` if any argument is missing, non-numeric, or index is invalid.
///
/// # Example
/// ```ignore
/// validate_1q_gate_3f4("u3", 3, &["0", "1.0", "2.0", "3.0"])?; // OK
/// validate_1q_gate_3f4("u3", 3, &["0", "NaN", "π", "0"])?;     // Error: invalid theta
/// ```
fn validate_1q_gate_3f4(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 4 {
        return Err("u3 requires 4 arguments: qubit, theta, phi, lambda".into());
    }

    let qubit = args[0].parse::<usize>()
        .map_err(|_| format!("u3: invalid qubit index '{}'", args[0]))?;
    
    check_bounds(name, qubit, total_qubits)?;

    args[1].parse::<f64>()
        .map_err(|_| format!("u3: invalid theta '{}'", args[1]))?;
    args[2].parse::<f64>()
        .map_err(|_| format!("u3: invalid phi '{}'", args[2]))?;
    args[3].parse::<f64>()
        .map_err(|_| format!("u3: invalid lambda '{}'", args[3]))?;

    Ok(())
}

/// Validates a two-qubit gate call (e.g., `cx(0, 1)`).
///
/// Checks:
/// - Exactly 2 arguments are provided
/// - Both are valid qubit indices and within bounds
/// - Indices refer to different qubits
///
/// # Parameters
/// - `name`: Gate name (for use in error messages)
/// - `total_qubits`: Number of available qubits
/// - `args`: Expected as `[control, target]`
///
/// # Returns
/// - `Ok(())` if validation succeeds
/// - `Err(String)` describing the issue otherwise
///
/// # Example
/// ```ignore
/// validate_2q_gate("cx", 4, &["0", "1"])?;    // OK
/// validate_2q_gate("cx", 4, &["0", "0"])?;    // Error: same qubit
/// ```

fn validate_2q_gate(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 2 {
        return Err(format!("{} requires 2 qubit indices", name));
    }

    let q0 = args[0].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[0]))?;
    let q1 = args[1].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[1]))?;

    check_bounds(name, q0, total_qubits)?;
    check_bounds(name, q1, total_qubits)?;

    if q0 == q1 {
        return Err(format!("{}: qubit indices must be different", name));
    }

    Ok(())
}

/// Validates a `controlled_u` gate with 2 qubit indices and 4 matrix parameters.
///
/// Expected input: `[control, target, u00, u01, u10, u11]`
///
/// Checks:
/// - 6 arguments are provided
/// - Qubit indices are valid and distinct
/// - All matrix parameters are valid floats
///
/// # Parameters
/// - `name`: Name of the gate (typically `"controlled_u"` or `"cu"`)
/// - `total_qubits`: Number of qubits in the QVM
/// - `args`: Argument list `[control, target, u00, u01, u10, u11]`
///
/// # Returns
/// - `Ok(())` if validation passes
/// - `Err(String)` with a detailed error message otherwise
///
/// # Example
/// ```ignore
/// validate_controlled_u("cu", 4, &["0", "1", "1.0", "0.0", "0.0", "1.0"])?; // OK
/// validate_controlled_u("cu", 4, &["0", "0", "1.0", "x", "0", "1"])?;       // Error
/// ```

fn validate_controlled_u(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 6 {
        return Err(format!("{} requires 6 arguments: control, target, u00, u01, u10, u11", name));
    }

    let q0 = args[0].parse::<usize>()
        .map_err(|_| format!("{}: invalid control qubit '{}'", name, args[0]))?;

    let q1 = args[1].parse::<usize>()
        .map_err(|_| format!("{}: invalid target qubit '{}'", name, args[1]))?;

    check_bounds(name, q0, total_qubits)?;
    check_bounds(name, q1, total_qubits)?;

    if q0 == q1 {
        return Err(format!("{}: control and target must be different", name));
    }

    for (i, val) in args[2..].iter().enumerate() {
        val.parse::<f64>()
            .map_err(|_| format!("{}: invalid matrix parameter at position {}: '{}'", name, i + 2, val))?;
    }

    Ok(())
}

/// Validates a three-qubit gate call (e.g., `toffoli(0, 1, 2)`).
///
/// Checks:
/// - Exactly 3 arguments are provided
/// - All are valid and distinct qubit indices
/// - All are within bounds
///
/// # Parameters
/// - `name`: Gate name (e.g., "toffoli", "fredkin")
/// - `total_qubits`: Total number of qubits available
/// - `args`: Expected as `[q0, q1, q2]`
///
/// # Returns
/// - `Ok(())` if all indices are valid and unique
/// - `Err(String)` if any check fails
///
/// # Example
/// ```ignore
/// validate_3q_gate("toffoli", 5, &["0", "1", "2"])?;  // OK
/// validate_3q_gate("toffoli", 5, &["0", "1", "1"])?;  // Error: duplicate qubits
/// ```

fn validate_3q_gate(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 3 {
        return Err(format!("{} requires 3 qubit indices", name));
    }

    let q0 = args[0].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[0]))?;
    let q1 = args[1].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[1]))?;
    let q2 = args[2].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[2]))?;

    if q0 == q1 || q0 == q2 || q1 == q2 {
        return Err(format!("{}: all qubit indices must be distinct", name));
    }

    check_bounds(name, q0, total_qubits)?;
    check_bounds(name, q1, total_qubits)?;
    check_bounds(name, q2, total_qubits)?;

    Ok(())
}

/// Validates whether a qubit index is within the valid range of the quantum system.
///
/// # Parameters
/// - `name`: The name of the gate or operation performing the check (used in error messages).
/// - `q`: The qubit index being accessed.
/// - `total`: The total number of available qubits (e.g., from the QVM).
///
/// # Returns
/// - `Ok(())` if `q` is a valid index (i.e., `q < total`)
/// - `Err(String)` with a descriptive error message if the index is out of bounds.
///
///
/// # Error Message Format
/// `"cx: qubit index 5 out of bounds (max is 3)"`
/// /// # Example (for internal use)
/// ```ignore
/// check_bounds("cx", 2, 4)?; // valid
/// check_bounds("h", 5, 4)?;  // error: qubit 5 out of bounds
/// ```
fn check_bounds(name: &str, q: usize, total: usize) -> Result<(), String> {
    if q >= total {
        Err(format!("{}: qubit index {} out of bounds (max is {})", name, q, total - 1))
    } else {
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_validate_valid_1q_gate() {
        let args = vec!["0"];
        assert!(validate_1q_gate("x", 2, &args).is_ok());
    }

    #[test]
    fn test_validate_invalid_1q_gate_index() {
        let args = vec!["5"];
        assert!(validate_1q_gate("x", 2, &args).is_err());
    }

    #[test]
    fn test_validate_controlled_u_valid() {
        let args = vec!["0", "1", "1.0", "0.0", "0.0", "1.0"];
        assert!(validate_controlled_u("cu", 2, &args).is_ok());
    }

    #[test]
    fn test_validate_controlled_u_invalid_matrix() {
        let args = vec!["0", "1", "NaN", "?", "0", "1"];
        assert!(validate_controlled_u("cu", 2, &args).is_err());
    }

}