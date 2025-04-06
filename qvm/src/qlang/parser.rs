use crate::qlang::{ast::QLangCommand, aliases::resolve_alias};
use regex::Regex;

#[derive(Clone, Debug)]
pub enum QLangLine {
    Command(QLangCommand),
    Run,
    Reset,
}

pub struct QLangParser {
    func_regex: Regex,
}

impl QLangParser {
    pub fn new() -> Self {
        let func_regex = Regex::new(r"(\w+)\((.*)\)").unwrap();
        Self { func_regex }
    }

    pub fn parse_line(&self, line: &str) -> Result<QLangLine, String> {
        let caps = self.func_regex.captures(line.trim()).ok_or("Invalid syntax")?;
        let raw = caps.get(1).ok_or("Invalid function name")?.as_str();
        let args_str = caps.get(2).ok_or("Invalid argument list")?.as_str();

        let args: Vec<String> = args_str
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();

        let canonical = resolve_alias(raw);

        match canonical {
            "run" => Ok(QLangLine::Run),
            "reset" => Ok(QLangLine::Reset),
            "create" => self.parse_create(args).map(QLangLine::Command),
            "display" => Ok(QLangLine::Command(QLangCommand::Display)),
            "measure" => self.parse_measure(args).map(QLangLine::Command),
            "measure_all" => Ok(QLangLine::Command(QLangCommand::MeasureAll)),
            _ => self.parse_gate(canonical.to_string(), args).map(QLangLine::Command),
        }
    }

    fn parse_create(&self, args: Vec<String>) -> Result<QLangCommand, String> {
        let n = args.get(0)
            .ok_or("Missing argument for create")?
            .parse::<usize>()
            .map_err(|_| "Invalid number in create")?;
        Ok(QLangCommand::Create(n))
    }

    fn parse_measure(&self, args: Vec<String>) -> Result<QLangCommand, String> {
        if args.is_empty() {
            Ok(QLangCommand::MeasureAll)
        } else {
            let qubits: Result<Vec<usize>, _> = args.iter()
                .map(|a| a.parse::<usize>().map_err(|_| format!("Invalid qubit: {}", a)))
                .collect();
            Ok(QLangCommand::MeasureMany(qubits?))
        }
    }

    fn parse_gate(&self, name: String, args: Vec<String>) -> Result<QLangCommand, String> {
        Ok(QLangCommand::ApplyGate(name, args))
    }
}
