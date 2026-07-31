use std::collections::HashMap;

use qlang::{
    ast::{Expression, QLangCommand},
    interpreter::{run_ast, ControlFlow},
    QLang,
};

#[test]
fn measurement_alias_is_parsed_as_measurement() {
    let mut qlang = QLang::new(1);
    qlang.append_from_str("m(0)");
    qlang.parser.validate_lines();

    assert!(!qlang.parser.has_errors());
    assert!(matches!(
        qlang.parser.get_commands().first(),
        Some(qlang::parser::QLangLine::Command(QLangCommand::Measure(0)))
    ));
}

#[test]
fn parser_errors_are_returned_to_the_caller() {
    let mut qlang = QLang::new(1);
    qlang.append_from_str("h(");

    let error = qlang.run_parsed_commands().unwrap_err();
    assert!(!error.is_empty());
}

#[test]
fn create_reconfigures_the_quantum_register() {
    let mut qlang = QLang::new(1);
    qlang.append_from_str("create(3)");
    qlang.run_parsed_commands().unwrap();

    assert_eq!(qlang.qvm.num_qubits(), 3);
    assert!(matches!(qlang.ast.as_slice(), [QLangCommand::Create(3)]));
}

#[test]
fn run_from_str_executes_measurement_only_once() {
    let mut qlang = QLang::new(1);
    qlang.run_from_str("x(0)\nmeasure(0)\n");

    let state = qlang.qvm.state_vector();
    assert!(state[0].norm_sqr() < 1e-12);
    assert!((state[1].norm_sqr() - 1.0).abs() < 1e-12);
}

#[test]
fn runtime_reports_duplicate_qubits_without_panicking() {
    let mut qvm = qvm::QVM::new(2);
    let result = run_ast(
        &mut qvm,
        &[QLangCommand::ApplyGate(
            "cnot".into(),
            vec![Expression::Number(0.0), Expression::Number(0.0)],
        )],
        &mut HashMap::new(),
        &mut HashMap::new(),
    );

    assert!(matches!(
        result,
        ControlFlow::Error(error) if error.contains("must be different")
    ));
}

#[test]
fn runtime_rejects_non_gates_in_qif_and_restores_else_control() {
    let mut qvm = qvm::QVM::new(2);
    let result = run_ast(
        &mut qvm,
        &[QLangCommand::QIf {
            condition: Expression::Number(0.0),
            then_branch: vec![],
            else_branch: Some(vec![QLangCommand::Let {
                name: "value".into(),
                type_ann: None,
                value: Expression::Number(1.0),
            }]),
        }],
        &mut HashMap::new(),
        &mut HashMap::new(),
    );

    assert!(matches!(result, ControlFlow::Error(_)));
    let state = qvm.state_vector();
    assert!((state[0].re - 1.0).abs() < 1e-12);
    assert!(state[1].norm_sqr() < 1e-12);
}
