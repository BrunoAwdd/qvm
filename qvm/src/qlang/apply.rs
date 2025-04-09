use crate::qvm::QVM;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// Porta de 1 qubit sem parâmetro
pub fn apply_one_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q = parse_usize(&args[0]);
    qvm.apply_gate(gate, q);
}

/// Porta de 1 qubit com 1 parâmetro f64
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

/// Porta de 1 qubit com 2 parâmetros f64
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

/// Porta de 1 qubit com 3 parâmetros f64
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

/// Porta de 2 qubits
pub fn apply_two_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q0 = parse_usize(&args[0]);
    let q1 = parse_usize(&args[1]);
    qvm.apply_gate_2q(gate, q0, q1);
}

/// Porta CU (Controlled-U) com 4 parâmetros f64 (reais ou componentes da matriz)
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


/// Porta de 3 qubits
pub fn apply_three_q_gate<G: QuantumGateAbstract>(qvm: &mut QVM, gate: &G, args: &[String]) {
    let q0 = parse_usize(&args[0]);
    let q1 = parse_usize(&args[1]);
    let q2 = parse_usize(&args[2]);
    qvm.apply_gate_3q(gate, q0, q1, q2);
}

/// Auxiliares
fn parse_usize(s: &str) -> usize {
    s.parse::<usize>().unwrap_or_else(|_| panic!("Qubit inválido: '{}'", s))
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or_else(|_| panic!("Parâmetro inválido: '{}'", s))
}
