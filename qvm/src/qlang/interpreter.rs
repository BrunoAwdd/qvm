use crate::qvm::QVM;
use crate::qlang::{ast::QLangCommand, apply::*};
use crate::gates::{
    one_q::{hadamard::*, identity::*, pauli_x::*, pauli_y::*, pauli_z::*, s::*, s_dagger::*, t::*, t_dagger::*},
    rotation_q::{phase::*, rx::*, ry::*, rz::*, u1::*, u2::*, u3::*},
    two_q::{cnot::*, cy::*, cz::*, iswap::*, swap::*, },
    three_q::{fredkin::*, toffoli::*},
};

/// Executes a sequence of QLang commands (AST) on a given quantum virtual machine.
///
/// This function interprets the `QLangCommand` values and dispatches
/// them to the appropriate quantum gate implementations.
///
/// # Parameters
/// - `qvm`: A mutable reference to the quantum virtual machine.
/// - `ast`: A list of parsed QLang commands.
///
/// # Behavior
/// - `Create(n)`: Reinitializes the QVM with `n` qubits.
/// - `ApplyGate(name, args)`: Applies the specified gate to the QVM.
/// - `Display`: Triggers a debug/state display from the QVM.
/// - `MeasureAll` / `Measure` / `MeasureMany`: Performs measurements.
///
/// Unknown gate names are printed as warnings.
pub fn run_ast(qvm: &mut QVM, ast: &[QLangCommand]) {
    for cmd in ast {
        match cmd {
            QLangCommand::Create(n) => {
                *qvm = QVM::new(*n);
            }
            QLangCommand::ApplyGate(name, args) => {
                apply_gate_dispatch(qvm, name, args);
            },
            QLangCommand::Display => qvm.display(),
            QLangCommand::MeasureAll => { qvm.measure_all(); },
            QLangCommand::Measure(q) => { qvm.measure(*q); },
            QLangCommand::MeasureMany(qs) => { qvm.measure_many(qs);},
        }
    }
}

/// Dispatches a gate application by matching the gate name to its implementation.
///
/// This function maps a gate name (and its optional aliases) to the appropriate
/// quantum gate logic and applies it to the quantum virtual machine (`QVM`)
/// using the given arguments.
///
/// # Parameters
/// - `qvm`: A mutable reference to the quantum virtual machine.
/// - `name`: The canonical or aliased name of the gate to apply.
/// - `args`: A list of string arguments passed to the gate. These are typically qubit indices,
///           and possibly parameters like angles for rotation gates.
///
/// # Behavior
/// - For single-qubit gates (e.g., `h`, `x`, `rz`), dispatches to `apply_one_q_gate` or its variants.
/// - For two- and three-qubit gates (e.g., `cx`, `toffoli`), dispatches accordingly.
/// - For gates requiring parameters (e.g., `rx`, `u3`), converts arguments as needed.
///
/// # Panics / Errors
/// This function does not return errors, but:
/// - If `args` are invalid (e.g., wrong number of arguments, parse failure),
///   the called gate function is responsible for handling it (often via `Result` or internal logging).
///
/// # Notes
/// - Unknown or unsupported gate names are logged via `println!`.
/// - Aliases like `"h"` → `"hadamard"` and `"cx"` → `"cnot"` are supported.
fn apply_gate_dispatch(qvm: &mut QVM, name: &str, args: &[String]) {
    match name {
        "controlled_u" | "cu" => apply_controlled_u(qvm, args),
        "hadamard" | "h"    => apply_one_q_gate(qvm, &Hadamard::new(), args),
        "identity" | "id"   => apply_one_q_gate(qvm, &Identity::new(), args),
        "paulix" | "x"      => apply_one_q_gate(qvm, &PauliX::new(), args),
        "pauliy" | "y"      => apply_one_q_gate(qvm, &PauliY::new(), args),
        "pauliz" | "z"      => apply_one_q_gate(qvm, &PauliZ::new(), args),
        "s"                 => apply_one_q_gate(qvm, &S::new(), args),
        "sdagger" | "sdg"   => apply_one_q_gate(qvm, &SDagger::new(), args),
        "t"                 => apply_one_q_gate(qvm, &T::new(), args),
        "tdagger" | "tdg"   => apply_one_q_gate(qvm, &TDagger::new(), args),
        "phase"             => apply_one_q_with_1f64(qvm, &Phase::new, args),
        "rx"                => apply_one_q_with_1f64(qvm, &RX::new, args),
        "ry"                => apply_one_q_with_1f64(qvm, &RY::new, args),
        "rz"                => apply_one_q_with_1f64(qvm ,&RZ::new, args),
        "u1"                => apply_one_q_with_1f64(qvm, &U1::new, args),
        "u2"                => apply_one_q_with_2f64(qvm, &U2::new, args),
        "u3"                => apply_one_q_with_3f64(qvm, &U3::new, args),
        "cnot" | "cx"       => apply_two_q_gate(qvm, &CNOT::new(), args),
        "iswap"             => apply_two_q_gate(qvm, &ISwap::new(), args),
        "swap"              => apply_two_q_gate(qvm, &Swap::new(), args),
        "cy"                => apply_two_q_gate(qvm, &ControlledY::new(), args),
        "cz"                => apply_two_q_gate(qvm, &ControlledZ::new(), args),
        "toffoli"           => apply_three_q_gate(qvm, &Toffoli::new(), args),
        "fredkin"           => apply_three_q_gate(qvm, &Fredkin::new(), args),  
        _ => println!("Gate desconhecido: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qvm::QVM;

    #[test]
    fn test_hadamard_application() {
        let mut qvm = QVM::new(1);
        let ast = vec![
            QLangCommand::Create(1),
            QLangCommand::ApplyGate("hadamard".into(), vec!["0".into()])
        ];
        run_ast(&mut qvm, &ast);
        
        let state = qvm.state_vector();
        let norm = (1.0 / 2.0f64).sqrt();
        assert!((state[0].norm_sqr() - norm.powi(2)).abs() < 1e-6);
        assert!((state[1].norm_sqr() - norm.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_measure_all() {
        let mut qvm = QVM::new(2);
        let ast = vec![
            QLangCommand::Create(2),
            QLangCommand::ApplyGate("x".into(), vec!["0".into()]),
            QLangCommand::MeasureAll
        ];
        run_ast(&mut qvm, &ast);
        
        let result = qvm.measure_all();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1); 
    }

    #[test]
    fn test_measure_many() {
        let mut qvm = QVM::new(3);
        let ast = vec![
            QLangCommand::Create(3),
            QLangCommand::ApplyGate("x".into(), vec!["2".into()]),
            QLangCommand::MeasureMany(vec![0, 2])
        ];
        run_ast(&mut qvm, &ast);
        
        let result = qvm.measure(2);
        assert_eq!(result, 1u8);
    }


}
