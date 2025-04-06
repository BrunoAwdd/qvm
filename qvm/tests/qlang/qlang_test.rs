use qlang::qlang::QLang;
use qlang::qlang::ast::QLangCommand;

#[test]
fn test_run_line_create() {
    let mut qlang = QLang::new(1);
    qlang.run_qlang_from_line("create(3)").unwrap();
    assert!(matches!(qlang.ast.last().unwrap(), QLangCommand::Create(3)));
}

#[test]
fn test_run_line_hadamard() {
    let mut qlang = QLang::new(1);
    qlang.run_qlang_from_line("h(0)").unwrap();
    assert!(matches!(qlang.ast.last().unwrap(), QLangCommand::ApplyGate(_, _)));
}

#[test]
fn test_run_line_measure() {
    let mut qlang = QLang::new(2);
    qlang.run_qlang_from_line("measure(1)").unwrap();
    assert!(matches!(qlang.ast.last().unwrap(), QLangCommand::MeasureMany(_)));
}

#[test]
fn test_to_source_reconstruction() {
    let mut qlang = QLang::new(1);
    qlang.run_qlang_from_line("x(0)").unwrap();
    qlang.run_qlang_from_line("measure(0)").unwrap();

    let src = qlang.to_source();
    assert!(src.contains("create(1)"));
    assert!(src.contains("paulix(0)"));
    assert!(src.contains("measure(0)"));
}

#[test]
fn test_reset_clears_ast() {
    let mut qlang = QLang::new(3);
    qlang.run_qlang_from_line("x(0)").unwrap();
    assert!(qlang.ast.len() > 0);

    qlang.reset();
    assert_eq!(qlang.ast.len(), 0);
}

