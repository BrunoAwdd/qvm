use crate::qvm::QVM;
use crate::qlang::{ast::QLangCommand, apply::*};
use crate::gates::{
    general::controlled_u::*,
    one_q::{hadamard::*, identity::*, pauli_x::*, pauli_y::*, pauli_z::*, s::*, s_dagger::*, t::*, t_dagger::*},
    rotation_q::{phase::*, rx::*, ry::*, rz::*, u1::*, u2::*, u3::*},
    two_q::{cnot::*, cy::*, cz::*, iswap::*, swap::*, },
    three_q::{fredkin::*, toffoli::*},
};


pub fn run_ast(qvm: &mut QVM, ast: &[QLangCommand]) {
    println!("Running AST with {:?} commands", ast);
    for cmd in ast {
        match cmd {
            QLangCommand::Create(n) => {
                *qvm = QVM::new(*n);
            }
            QLangCommand::ApplyGate(name, args) => {
                match name.as_str() {
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
                "cu"                => apply_controlled_u(qvm, args),
                "toffoli"           => apply_three_q_gate(qvm, &Toffoli::new(), args),
                "fredkin"           => apply_three_q_gate(qvm, &Fredkin::new(), args),  
                _ => println!("Gate desconhecido: {}", name),
            }},
            QLangCommand::Display => qvm.display(),
            QLangCommand::MeasureAll => { qvm.measure_all(); },
            QLangCommand::Measure(q) => { qvm.measure(*q); },
            QLangCommand::MeasureMany(qs) => for q in qs { qvm.measure(*q); },
        }
    }
}

