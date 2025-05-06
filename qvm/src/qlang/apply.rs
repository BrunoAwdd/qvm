use crate::qvm::QVM;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Applies a single-qubit gate with no parameters (e.g., `x(0)`, `h(1)`).
///
/// # Parameters
/// - `qvm`: The quantum virtual machine to apply the gate on.
/// - `gate`: The quantum gate instance to apply.
/// - `args`: A list of arguments; expected to contain one qubit index.
///
/// # Panics
/// Panics if the qubit index is not a valid `usize`.
pub fn apply_one_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q = parse_usize(&args[0]);
    qvm.apply_gate(gate, q);
}

/// Applies a single-qubit gate that takes one float parameter (e.g., `rx(0, θ)`).
///
/// # Parameters
/// - `qvm`: The quantum virtual machine.
/// - `constructor`: A closure to construct the gate with one `f64` parameter.
/// - `args`: A list of arguments; expected: `[qubit, param]`.
///
/// # Panics
/// Panics if argument parsing fails.
pub fn apply_one_q_with_1f64<T, G>(qvm: &mut QVM, constructor: T, args: &[String])
where
    T: Fn(f64) -> G,
    G: QuantumGateAbstract,
{
    let q = parse_usize(&args[0]);
    let p = parse_f64(&args[1]);
    let gate = constructor(p);
    qvm.apply_gate(&gate, q);
}

/// Applies a single-qubit gate that takes two float parameters (e.g., `u2(0, φ, λ)`).
///
/// # Parameters
/// - `constructor`: Closure to build the gate with two `f64` values.
/// - `args`: `[qubit, phi, lambda]`
///
/// # Panics
/// Panics on invalid argument format or type.
pub fn apply_one_q_with_2f64<T, G>(qvm: &mut QVM, constructor: T, args: &[String])
where
    T: Fn(f64, f64) -> G,
    G: QuantumGateAbstract,
{
    let q = parse_usize(&args[0]);
    let p1 = parse_f64(&args[1]);
    let p2 = parse_f64(&args[2]);
    let gate = constructor(p1, p2);
    qvm.apply_gate(&gate, q);
}

/// Applies a single-qubit gate with three float parameters (e.g., `u3(0, θ, φ, λ)`).
///
/// # Parameters
/// - `constructor`: Function/closure that builds the gate from three floats.
/// - `args`: `[qubit, theta, phi, lambda]`
///
/// # Panics
/// Panics on invalid float or qubit argument.
pub fn apply_one_q_with_3f64<T, G>(qvm: &mut QVM, constructor: T, args: &[String])
where
    T: Fn(f64, f64, f64) -> G,
    G: QuantumGateAbstract,
{
    let q = parse_usize(&args[0]);
    let p1 = parse_f64(&args[1]);
    let p2 = parse_f64(&args[2]);
    let p3 = parse_f64(&args[3]);
    let gate = constructor(p1, p2, p3);
    qvm.apply_gate(&gate, q);
}

/// Applies a two-qubit gate (e.g., `cx(0, 1)`).
///
/// # Parameters
/// - `gate`: Two-qubit gate to apply.
/// - `args`: `[control, target]` qubit indices.
///
/// # Panics
/// Panics if qubit indices are invalid.
pub fn apply_two_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q0 = parse_usize(&args[0]);
    let q1 = parse_usize(&args[1]);
    qvm.apply_gate_2q(gate, q0, q1);
}

/// Applies a `Controlled-U` gate with real-valued matrix elements.
///
/// # Parameters
/// - `args`: `[control, target, u00, u01, u10, u11]`
///
/// # Panics
/// Panics if any argument is missing or cannot be parsed.
pub fn apply_controlled_u(qvm: &mut QVM, args: &[String]) {
    if args.len() != 6 {
        panic!("CU espera 6 argumentos: control, target, u00, u01, u10, u11");
    }

    let control = parse_usize(&args[0]);
    let target  = parse_usize(&args[1]);

    let u00 = parse_f64(&args[2]);
    let u01 = parse_f64(&args[3]);
    let u10 = parse_f64(&args[4]);
    let u11 = parse_f64(&args[5]);

    use crate::gates::general::controlled_u::ControlledU;

    let gate = ControlledU::new_real(u00, u01, u10, u11); // ou .new_complex se fizer isso depois
    qvm.apply_gate_2q(&gate, control, target);
}


/// Applies a three-qubit gate (e.g., `toffoli(0, 1, 2)`).
///
/// # Parameters
/// - `gate`: A three-qubit gate.
/// - `args`: `[q0, q1, q2]` — the qubit indices.
///
/// # Panics
/// Panics if any qubit index is invalid.
pub fn apply_three_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q0 = parse_usize(&args[0]);
    let q1 = parse_usize(&args[1]);
    let q2 = parse_usize(&args[2]);
    qvm.apply_gate_3q(gate, q0, q1, q2);
}

/// Parses a `&str` into `usize`, panicking with context on failure.
fn parse_usize(s: &str) -> usize {
    s.parse::<usize>().unwrap_or_else(|_| panic!("Qubit inválido: '{}'", s))
}

/// Parses a `&str` into `f64`, panicking with context on failure.
fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or_else(|_| panic!("Parâmetro inválido: '{}'", s))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::qvm::QVM;
    use crate::gates::one_q::pauli_x::PauliX;

    fn str_args(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_apply_one_q_gate_valid() {
        let mut qvm = QVM::new(1);
        let gate = PauliX::new();
        apply_one_q_gate(&mut qvm, &gate, &str_args(&["0"]));
        // We assume no panic = success (behavior depends on QVM internals)
    }

    #[test]
    #[should_panic(expected = "Qubit inválido")]
    fn test_apply_one_q_gate_invalid_qubit() {
        let mut qvm = QVM::new(1);
        let gate = PauliX::new();
        apply_one_q_gate(&mut qvm, &gate, &str_args(&["not_a_number"]));
    }

    #[test]
    fn test_apply_one_q_with_1f64_valid() {
        let mut qvm = QVM::new(1);
        apply_one_q_with_1f64(&mut qvm, |theta| crate::gates::rotation_q::rx::RX::new(theta), &str_args(&["0", "3.14"]));
    }

    #[test]
    #[should_panic(expected = "Parâmetro inválido")]
    fn test_apply_one_q_with_1f64_invalid_param() {
        let mut qvm = QVM::new(1);
        apply_one_q_with_1f64(&mut qvm, |theta| crate::gates::rotation_q::rx::RX::new(theta), &str_args(&["0", "π"]));
    }

    #[test]
    fn test_apply_two_q_gate_valid() {
        let mut qvm = QVM::new(2);
        let gate = crate::gates::two_q::cnot::CNOT::new();
        apply_two_q_gate(&mut qvm, &gate, &str_args(&["0", "1"]));
    }

    #[test]
    fn test_apply_controlled_u_valid() {
        let mut qvm = QVM::new(2);
        apply_controlled_u(&mut qvm, &str_args(&["0", "1", "1.0", "0.0", "0.0", "1.0"]));
    }

    #[test]
    #[should_panic(expected = "CU espera 6 argumentos")]
    fn test_apply_controlled_u_invalid_args() {
        let mut qvm = QVM::new(2);
        apply_controlled_u(&mut qvm, &str_args(&["0", "1"]));
    }

    #[test]
    fn test_apply_three_q_gate_valid() {
        let mut qvm = QVM::new(3);
        let gate = crate::gates::three_q::toffoli::Toffoli::new();
        apply_three_q_gate(&mut qvm, &gate, &str_args(&["0", "1", "2"]));
    }

    #[test]
    fn test_apply_two_qubit_gate() {
        let mut qvm = QVM::new(2);
        let gate = crate::gates::two_q::cnot::CNOT::new();
        apply_two_q_gate(&mut qvm, &gate, &["0".into(), "1".into()]);
    }
}
