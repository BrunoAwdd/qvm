use qlang::qlang::QLang;
use qlang::qlang::ast::QLangCommand;

#[test]
fn test_run_line_hadamard() {
    let mut qlang = QLang::new(1);
    qlang.run_qlang_from_line("h(0)").unwrap();
    assert!(matches!(qlang.ast.last().unwrap(), QLangCommand::ApplyGate(_, _)));
}

#[test]
fn test_run_line_measure() {
    let mut qlang = QLang::new(2);
    let result = qlang.run_qlang_from_line("measure(1)").unwrap();

    assert!(result.is_some());

    let measurements = result.unwrap();
    assert_eq!(measurements.len(), 1);
    assert!(matches!(measurements[0], 0 | 1));
}

#[test]
fn test_to_source_reconstruction() {
    let mut qlang = QLang::new(1);
    qlang.run_qlang_from_line("x(0)").unwrap();

    let src = qlang.to_source(); // <-- aqui!
    assert!(src.contains("create(1)"));
    assert!(src.contains("paulix(0)"));

    qlang.run_qlang_from_line("measure(0)").unwrap();
}
#[test]
fn test_reset_clears_ast() {
    let mut qlang = QLang::new(3);
    qlang.run_qlang_from_line("x(0)").unwrap();
    assert!(qlang.ast.len() > 0);

    qlang.reset();
    assert_eq!(qlang.ast.len(), 0);
}

