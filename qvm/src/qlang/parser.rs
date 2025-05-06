use regex::Regex;
use std::fmt;

use super::{aliases::resolve_alias, ast::QLangCommand};

/// Represents a single parsed line in the QLang program.
#[derive(Clone, Debug)]
pub enum QLangLine {
    /// A structured command (e.g., `Create`, `ApplyGate`, etc.).
    Command(QLangCommand),

    /// Special instruction to execute the current quantum program.
    Run,

    /// Special instruction to reset the quantum virtual machine.
    Reset,
}

impl fmt::Display for QLangLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangLine::Command(cmd) => write!(f, "{cmd}"),
            QLangLine::Run => write!(f, "RUN"),
            QLangLine::Reset => write!(f, "RESET"),
        }
    }
}
/// Parses and validates QLang source code into an abstract syntax tree (AST).
/// 
/// Stores raw input lines, collects syntax errors, and produces structured commands.
#[derive(Clone)]
pub struct QLangParser {
    func_regex: Regex,
    errors: Vec<String>,
    commands: Vec<String>,
    parsed_commands: Vec<QLangLine>,
}

impl QLangParser {
    /// Creates a new `QLangParser` with default configuration and regex pattern.
    pub fn new() -> Self {
        let func_regex = Regex::new(r"(\w+)\((.*)\)")
            .expect("QLangParser: Invalid regex: (\\w+)\\((.*)\\)");

        Self { func_regex, errors: vec![], commands: vec![], parsed_commands: vec![] }
    }
    /// Appends a raw line of QLang source code to be parsed later.
    ///
    /// The line is stored and will be processed during `validate_lines()`.
    ///
    /// # Example
    /// ```
    /// use qlang::qlang::parser::QLangParser;
    /// let mut parser = QLangParser::new();
    /// parser.append("create(3)");
    /// ```
    pub fn append(&mut self, line: &str){
        self.commands.push(line.to_string());
    }

    /// Parses all appended source lines and converts them into structured commands.
    ///
    /// On success, parsed commands will be available via `get_commands()`.
    /// Any syntax errors will be stored in `get_errors()`.
    ///
    /// This method clears previously parsed data and errors.
    pub fn validate_lines(&mut self) {
        self.errors.clear();
        self.parsed_commands.clear();

        for line in self.commands.iter() {
            match self.parse_line(line) {
                Ok(parsed) => self.parsed_commands.push(parsed),
                Err(err) => self.errors.push(format!("Erro na linha '{}': {}", line, err)),
            }
        }
    }

    /// Returns `true` if any syntax errors were encountered during parsing.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns a list of error messages from the most recent validation.
    pub fn get_errors(&self) -> &[String] {
        &self.errors
    }

    /// Returns a list of error messages from the most recent validation.
    pub fn get_commands(&self) -> &[QLangLine] {
        &self.parsed_commands
    }

    /// Attempts to parse a single line of QLang source code into a `QLangLine`.
    ///
    /// Returns an error if the syntax is invalid or arguments are malformed.
    fn parse_line(&self, line: &str) -> Result<QLangLine, String> {
        let caps = self.func_regex.captures(line.trim()).ok_or("Invalid syntax")?;
        let (raw, args_str) = (
            caps.get(1).map(|m| m.as_str()).ok_or("Invalid function name")?,
            caps.get(2).map(|m| m.as_str()).ok_or("Invalid argument list")?,
        );

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

    /// Parses the arguments of a `create(...)` command into a `QLangCommand::Create`.
    ///
    /// Expects a single numeric argument representing the number of qubits.
    ///
    /// # Errors
    /// Returns an error if:
    /// - No argument is provided
    /// - The argument is not a valid `usize`   
    fn parse_create(&self, args: Vec<String>) -> Result<QLangCommand, String> {
        let n = args.get(0)
            .ok_or("Missing argument for create")?
            .parse::<usize>()
            .map_err(|_| "Invalid number in create")?;
        Ok(QLangCommand::Create(n))
    }

    /// Parses the arguments of a `measure(...)` command into a `QLangCommand`.
    ///
    /// - If no arguments are provided, it returns `MeasureAll`.
    /// - If arguments are present, it attempts to parse them as qubit indices for `MeasureMany`.
    ///
    /// # Errors
    /// Returns an error if any argument is not a valid `usize`.
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

    /// Parses a gate application command like `x(0)` or `cx(0, 1)`.
    ///
    /// Converts the command name and its arguments into a `QLangCommand::ApplyGate`.
    /// Arguments are passed as raw strings to preserve flexibility (e.g., for symbolic gates).
    fn parse_gate(&self, name: String, args: Vec<String>) -> Result<QLangCommand, String> {
        Ok(QLangCommand::ApplyGate(name, args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_create() {
        let mut parser = QLangParser::new();
        let lines = "create(4)";
        parser.append(&lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::Create(4).to_string());
    }

    #[test]
    fn parse_hadamard() {
        let mut parser = QLangParser::new();
        let lines = "h(0)";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::ApplyGate("hadamard".into(), vec!["0".into()]).to_string());
    }

    #[test]
    fn parse_rx() {
        let mut parser = QLangParser::new();
        let lines = "rx(1,3.14)";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::ApplyGate("rx".into(), vec!["1".into(), "3.14".into()]).to_string());
    }

    #[test]
    fn parse_measure_all() {
        let mut parser = QLangParser::new();
        let lines = "measure()";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::MeasureAll.to_string());
    }

    #[test]
    fn parse_measure_single() {
        let mut parser = QLangParser::new();
        let lines = "measure(2)";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::MeasureMany(vec![2]).to_string());
    }

    #[test]
    fn parse_measure_many() {
        let mut parser = QLangParser::new();
        let lines = "measure(1,2,3)";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::MeasureMany(vec![1, 2, 3]).to_string());
    }

    #[test]
    fn parse_display() {
        let mut parser = QLangParser::new();
        let lines = "display()";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::Display.to_string());
    }

    #[test]
    fn parse_unknown_gate() {
        let mut parser = QLangParser::new();
        let lines = "unknown(1,2)";
        parser.append(lines);
        parser.validate_lines();
        let cmd = parser.get_commands();
        assert_eq!(cmd[0].to_string(), QLangCommand::ApplyGate("unknown".into(), vec!["1".into(), "2".into()]).to_string());
    }


}