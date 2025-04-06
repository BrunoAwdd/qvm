use qlang::qlang::parser::QLangParser;
use qlang::qlang::ast::QLangCommand;

#[test]
fn parse_create() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("create(4)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::Create(4).to_string());
}

#[test]
fn parse_hadamard() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("h(0)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::ApplyGate("hadamard".into(), vec!["0".into()]).to_string());
}

#[test]
fn parse_rx() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("rx(1,3.14)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::ApplyGate("rx".into(), vec!["1".into(), "3.14".into()]).to_string());
}

#[test]
fn parse_measure_all() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("measure()").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::MeasureAll.to_string());
}

#[test]
fn parse_measure_single() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("measure(2)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::MeasureMany(vec![2]).to_string());
}

#[test]
fn parse_measure_many() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("measure(0,1,3)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::MeasureMany(vec![0, 1, 3]).to_string());
}

#[test]
fn parse_display() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("display()").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::Display.to_string());
}

#[test]
fn parse_unknown_gate() {
    let parser = QLangParser::new();
    let cmd = parser.parse_line("unknown(1,2)").unwrap();
    assert_eq!(cmd.to_string(), QLangCommand::ApplyGate("unknown".into(), vec!["1".into(), "2".into()]).to_string());
}
