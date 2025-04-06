pub fn validate_gate_arity(name: &str, total_qubits: usize, args: &[&str], ) -> Result<(), String>{
    
    match name {
        "hadamard" | "identity" | "paulix" | "pauliy" | 
        "pauliz" | "s" | "s_dagger" | "t" | "t_dagger" =>  validate_1q_gate(name, total_qubits, args),
        "u1" => validate_1q_gate_1f4(name, total_qubits, args),
        "u2" => validate_1q_gate_2f4(name, total_qubits, args),
        "u3" => validate_1q_gate_3f4(name, total_qubits, args),
        "rx" | "ry" | "rz" => validate_1q_gate_1f4(name, total_qubits, args),
        "cnot" | "swap" => validate_2q_gate(name, total_qubits, args),
        "toffoli" | "fredkin" => validade_3q_gate(name, total_qubits, args),
        _ => Err(format!("Unknown gate: '{}'", name)),
    }
}

fn validate_1q_gate(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 qubit indices", name));
    }

    let qubit = args[0].parse::<usize>().map_err(|_| format!("Invalid qubit index: {}", args[0]))?;
    check_bounds(name, qubit, total_qubits)?;

    Ok(())
}

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

fn validade_3q_gate(name: &str, total_qubits: usize, args: &[&str]) -> Result<(), String> {
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

fn check_bounds(name: &str, q: usize, total: usize) -> Result<(), String> {
    if q >= total {
        Err(format!("{}: qubit index {} out of bounds (max is {})", name, q, total - 1))
    } else {
        Ok(())
    }
}