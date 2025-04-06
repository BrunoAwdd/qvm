use qlang::qlang::ast::QLangCommand;

#[test]
fn display_create() {
    let cmd = QLangCommand::Create(3);
    assert_eq!(cmd.to_string(), "create(3)\n");
}

#[test]
fn display_apply_gate() {
    let cmd = QLangCommand::ApplyGate("h".into(), vec!["0".into()]);
    assert_eq!(cmd.to_string(), "h(0)\n");

    let cmd = QLangCommand::ApplyGate("rx".into(), vec!["0".into(), "3.14".into()]);
    assert_eq!(cmd.to_string(), "rx(0,3.14)\n");
}

#[test]
fn display_measure_all() {
    let cmd = QLangCommand::MeasureAll;
    assert_eq!(cmd.to_string(), "measure_all()\n");
}

#[test]
fn display_measure_single() {
    let cmd = QLangCommand::Measure(2);
    assert_eq!(cmd.to_string(), "measure(2)\n");
}

#[test]
fn display_measure_many() {
    let cmd = QLangCommand::MeasureMany(vec![0, 1, 3]);
    assert_eq!(cmd.to_string(), "measure(0,1,3)\n");
}

#[test]
fn display_display() {
    let cmd = QLangCommand::Display;
    assert_eq!(cmd.to_string(), "display()\n");
}
