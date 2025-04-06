use qlang::qlang::ast::QLangCommand;
use qlang::qlang::interpreter::run_ast;
use qlang::qvm::QVM;

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
