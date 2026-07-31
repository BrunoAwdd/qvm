use qlang::{ast::QLangCommand, QLang};

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
