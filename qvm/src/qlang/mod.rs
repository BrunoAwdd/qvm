pub mod aliases;
pub mod apply;
pub mod ast;
pub mod errors;
pub mod parser;
pub mod interpreter;
pub mod validators;

use ast::AstController;
use regex::Regex;
use std::fs;
use std::str::Lines;

use crate::qlang::{ast::QLangCommand, parser::{QLangParser, QLangLine}, interpreter::run_ast, validators::validate_gate_arity};
use crate::qvm::{QVM, backend::QuantumBackend};

/// The main interpreter and runtime for the QLang quantum programming language.
///
/// This struct is responsible for:
/// - Parsing source code into AST (`QLangCommand`)
/// - Validating syntax and argument correctness
/// - Interfacing with the quantum virtual machine (QVM)
/// - Executing quantum instructions
pub struct QLang {
    /// The underlying quantum virtual machine that executes commands.
    pub qvm: QVM,

    /// The abstract syntax tree built from parsed QLang commands.
    pub ast: Vec<QLangCommand>,

    /// The abstract syntax tree built from parsed QLang commands only for Controller.
    pub ast_controller: AstController,

    /// Flag to indicate that a `measure_all` has been issued (used to collapse execution).
    pub collapsed: bool,

    /// The parser that handles syntax and command resolution.
    pub parser: QLangParser,

    /// Regex used to extract function names and arguments from lines.
    func_regex: Regex,
}

impl Clone for QLang {
    fn clone(&self) -> Self {
        Self {
            qvm: self.qvm.clone(),
            ast: self.ast.clone(),
            ast_controller: self.ast_controller.clone(),
            collapsed: self.collapsed,
            func_regex: Regex::new(self.func_regex.as_str()).unwrap(),
            parser: self.parser.clone(),
        }
    }
}

impl QLang {
    /// Creates a new `QLang` instance with the specified number of qubits.
    ///
    /// Initializes the QVM and inserts a `Create(n)` command into the AST.
    pub fn new(num_qubits: usize) -> Self {
        let qvm = QVM::new(num_qubits);
        let func_regex = Regex::new(r"(\w+)\((.*)\)").unwrap();
        Self {
            qvm,
            ast: vec![QLangCommand::Create(num_qubits)],
            ast_controller: AstController::new(num_qubits),
            collapsed: false,
            func_regex,
            parser: QLangParser::new(),
        }
    }

    /// Converts the current AST back into QLang source code.
    ///
    /// Each command is formatted using its `Display` implementation,
    /// and separated by newlines.
    pub fn to_source(&self) -> String {
        self.ast_controller.to_source()
    }
            
    /// Appends a command to the AST.
    ///
    /// If the command is `MeasureAll`, it marks the program as collapsed (executed).
    pub fn append(&mut self, cmd: QLangCommand) {
        if matches!(cmd, QLangCommand::MeasureAll) {
            self.collapsed = true;
        }
        self.push_ast(cmd);
    }

    /// Executes the current AST using the QVM.
    ///
    /// This function does not clear the AST after execution.
    pub fn run(&mut self) {
        run_ast(&mut self.qvm, &self.ast);
    }

    /// Parses and runs QLang code from a raw multi-line string.
    ///
    /// Lines are appended to the parser, validated, and executed.
    /// AST is also executed afterward.
    ///
    /// # Panics
    /// Panics if parsing fails or the code is malformed.
    pub fn run_from_str(&mut self, code: &str) {
        self.append_from_lines(code.lines());
        let _ =self.run_parsed_commands();

        self.run(); 
    }

    /// Validates and executes all commands previously parsed by the parser.
    ///
    /// This method:
    /// - Converts valid lines into AST
    /// - Executes instructions immediately if needed (e.g., measure, reset)
    /// - Returns measurement results if applicable
    ///
    /// # Returns
    /// - `Some(Vec<u8>)` if a measurement produced results
    /// - `Ok(None)` otherwise
    ///
    /// # Errors
    /// - If any gate fails arity validation or argument parsing.
    pub fn run_parsed_commands(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.parser.validate_lines();
        let commands = self.parser.get_commands().to_vec();
        let mut final_result: Option<Vec<u8>> = None;

        for cmd in commands {
            match cmd {
                QLangLine::Command(QLangCommand::ApplyGate(ref name, ref args)) => {
                    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                    validate_gate_arity(name, self.qvm.num_qubits(), &args_refs)?;
                    self.push_ast(QLangCommand::ApplyGate(name.clone(), args.clone()));
                }
                QLangLine::Command(QLangCommand::Create(n)) => {
                    if self.ast.iter().any(|cmd| matches!(cmd, QLangCommand::Create(_))) {
                        continue;
                    }
                    self.reset();
                    self.push_ast(QLangCommand::Create(n));
                }
                QLangLine::Command(QLangCommand::MeasureMany(qs)) => {
                    self.push_ast(QLangCommand::MeasureMany(qs.clone()));
                    self.run();
                    let results = self.qvm.measure_many(&qs);
                    self.clear_ast();
                    final_result = Some(results);
                }
                QLangLine::Command(QLangCommand::MeasureAll) => {
                    self.push_ast(QLangCommand::MeasureAll);
                    self.run();
                    let results = self.qvm.measure_all();
                    self.clear_ast();
                    final_result = Some(results);
                }
                QLangLine::Run => {
                    self.run();
                    self.clear_ast();
                }
                QLangLine::Reset => {
                    self.reset();
                    println!("Reset");
                }
                QLangLine::Command(cmd) => {
                    self.push_ast(cmd.clone());
                }
            }
        }

        Ok(final_result)
    }

    /// Loads, parses, and executes QLang source code from a file.
    ///
    /// # Panics
    /// Panics if the file cannot be read.
    pub fn run_qlang_from_file(&mut self, file_path: &str) {
        let program = fs::read_to_string(file_path)
            .expect("Erro ao ler o arquivo");
        
        self.append_from_lines(program.lines());
        let _ =self.run_parsed_commands();
    }

    /// Appends a single source line to the parser if it's not empty or a comment.
    ///
    /// Ignores lines that are blank or start with `//`.    
    pub fn append_from_str(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            return;
        }
        self.parser.append(line)
    }

    /// Appends multiple lines (from a string iterator) to the parser.
    ///
    /// This is typically used to prepare the code for validation and execution.
    pub fn append_from_lines(&mut self, lines:Lines ) {
        for line in lines {
            self.parser.append(line);
        }
    }

    /// Resets the QVM and AST, preserving the number of qubits.
    ///
    /// Also clears the collapsed flag.
    pub fn reset(&mut self) {
        let qubits = self.qvm.backend.num_qubits();
        self.qvm = QVM::new(qubits);
        self.clear_ast();
        self.collapsed = false;
    }

    /// Clears the current AST without affecting the QVM.
    pub fn clear_ast(&mut self) {
        self.ast.clear();
    }

    /// Internal helper to push a command to the AST.
    ///
    /// Not exposed publicly.
    fn push_ast(&mut self, cmd: QLangCommand) {
        self.ast.push(cmd.clone());
        self.ast_controller.append(&cmd);
    }

} 

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::qlang_complex::QLangComplex;

    #[test]
    fn test_run_line_hadamard() {
        let mut qlang = QLang::new(1);
        let lines = "h(0)";
        qlang.parser.append(lines);
        qlang.parser.validate_lines();
        let _ = qlang.run_parsed_commands();

        assert!(matches!(qlang.ast.last().unwrap(), QLangCommand::ApplyGate(_, _)));
    }

    #[test]
    fn test_run_line_measure() {
        let mut qlang = QLang::new(2);
        let lines = "measure(1)";
        qlang.parser.append(lines);
        qlang.parser.validate_lines();
        let result = qlang.run_parsed_commands().unwrap();

        assert!(result.is_some());

        let measurements = result.unwrap();
        assert_eq!(measurements.len(), 1);
        assert!(matches!(measurements[0], 0 | 1));
    }

    #[test]
    fn test_to_source_reconstruction() {
        let mut qlang = QLang::new(1);
        let lines = "x(0)";
        qlang.parser.append(lines);
        qlang.parser.validate_lines();

        qlang.run_parsed_commands().unwrap();

        let src = qlang.to_source();
        assert!(src.contains("create(1)"));
        assert!(src.contains("paulix(0)"));
    }
    #[test]
    fn test_reset_clears_ast() {
        let mut qlang = QLang::new(3);
        let lines = "x(0)";
        qlang.parser.append(lines);
        qlang.parser.validate_lines();
        assert!(qlang.ast.len() > 0);

        qlang.reset();
        assert_eq!(qlang.ast.len(), 0);
    }

    #[test]
    fn test_negative_complex() {
        let c = QLangComplex::neg_one();
        assert_eq!(c.re, -1.0);
        assert_eq!(c.im, 0.0);

        let result = c * QLangComplex::i(); // (-1 + 0i) * (0 + 1i) = 0 - i

        println!("{:?}", result);

        assert_eq!(result.re, 0.0);
        assert_eq!(result.im, -1.0);
    }

    #[test]
    fn test_parse_invalid_syntax() {
        let mut parser = QLangParser::new();
        parser.append("invalid line");
        parser.validate_lines();
        assert!(parser.has_errors());
    }

    #[test]
    fn test_parse_measure_all() {
        let mut parser = QLangParser::new();
        parser.append("measure_all()");
        parser.validate_lines();
        let cmds = parser.get_commands();
        assert!(matches!(cmds[0], QLangLine::Command(QLangCommand::MeasureAll)));
    }

    #[test]
    fn test_to_source_multiple_lines() {
        let mut qlang = QLang::new(2);
        let lines = "x(0)\nrx(1,3.14)\nmeasure_all()";
        qlang.append_from_lines(lines.lines());
        qlang.run_parsed_commands().unwrap();

        let src = qlang.to_source();

        println!("Src:\n{}", src);

        assert!(src.contains("create(2)"));
        assert!(src.contains("paulix(0)"));
        assert!(src.contains("rx(1,3.14)"));
        assert!(src.contains("measure_all()"));
    }
}