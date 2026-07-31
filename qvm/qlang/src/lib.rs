pub mod aliases;
pub mod apply;
pub mod ast;
pub mod batch;
pub mod errors;
pub mod interpreter;
pub mod parser;
pub mod type_checker;
pub mod validators;

use std::collections::HashMap;
use std::fs;
use std::str::Lines;

use crate::{
    ast::QLangCommand,
    interpreter::{run_ast, Value},
    parser::{QLangLine, QLangParser},
};
use qvm::QVM;

pub struct QLang {
    pub qvm: QVM,
    pub parser: QLangParser,
    pub ast: Vec<QLangCommand>,
    pub collapsed: bool,
    pub variables: HashMap<String, Value>,
    pub functions: HashMap<String, QLangCommand>,
}

impl Clone for QLang {
    fn clone(&self) -> Self {
        Self {
            qvm: self.qvm.clone(),
            ast: self.ast.clone(),
            collapsed: self.collapsed,
            parser: self.parser.clone(),
            variables: self.variables.clone(),
            functions: self.functions.clone(),
        }
    }
}

impl QLang {
    pub fn new(num_qubits: usize) -> Self {
        let qvm = QVM::new(num_qubits);
        Self {
            qvm,
            ast: vec![QLangCommand::Create(num_qubits)],
            collapsed: false,
            parser: QLangParser::new(),
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn to_source(&self) -> String {
        self.ast
            .iter()
            .map(|cmd| cmd.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn append(&mut self, cmd: QLangCommand) {
        if matches!(cmd, QLangCommand::MeasureAll) {
            self.collapsed = true;
        }
        self.ast.push(cmd);
    }

    /// Runs the type-checker first; aborts with typed errors if any are found.
    pub fn run(&mut self) {
        if let Err(error) = self.try_run() {
            eprintln!("{error}");
        }
    }

    pub fn try_run(&mut self) -> Result<(), String> {
        let mut tc = type_checker::TypeChecker::new(self.qvm.num_qubits());
        let errors = tc.check(&self.ast);
        if !errors.is_empty() {
            return Err(errors
                .into_iter()
                .map(|error| format!("[QLang type error] {error}"))
                .collect::<Vec<_>>()
                .join("\n"));
        }
        let _ = run_ast(
            &mut self.qvm,
            &self.ast,
            &mut self.variables,
            &mut self.functions,
        );
        Ok(())
    }

    pub fn run_from_str(&mut self, code: &str) {
        self.append_from_lines(code.lines());
        if let Err(error) = self.run_parsed_commands().and_then(|_| self.try_run()) {
            eprintln!("{error}");
        }
    }

    pub fn run_parsed_commands(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.parser.validate_lines();
        if self.parser.has_errors() {
            return Err(self.parser.get_errors().join("\n"));
        }
        let commands = self.parser.get_commands().to_vec();
        for cmd in commands {
            match cmd {
                QLangLine::Command(QLangCommand::Import { path }) => {
                    let file_path = if path == "std" {
                        "std.ql".to_string()
                    } else if path.ends_with(".ql") {
                        path.clone()
                    } else {
                        format!("{}.ql", path)
                    };
                    let content = fs::read_to_string(&file_path)
                        .map_err(|error| format!("Failed to import '{}': {}", path, error))?;
                    let mut parser = QLangParser::new();
                    for line in content.lines() {
                        parser.append(line);
                    }
                    parser.validate_lines();
                    if parser.has_errors() {
                        return Err(format!(
                            "Failed to import '{}': {}",
                            path,
                            parser.get_errors().join("\n")
                        ));
                    }
                    for imported in parser.get_commands() {
                        if let QLangLine::Command(command @ QLangCommand::FunctionDef { .. }) =
                            imported
                        {
                            self.push_ast(command.clone());
                        }
                    }
                }
                QLangLine::Command(QLangCommand::Create(n)) => {
                    self.qvm = QVM::new(n);
                    self.ast.clear();
                    self.variables.clear();
                    self.functions.clear();
                    self.collapsed = false;
                    self.push_ast(QLangCommand::Create(n));
                }
                QLangLine::Run => {
                    self.try_run()?;
                    self.clear_ast();
                }
                QLangLine::Reset => {
                    self.reset();
                    println!("Reset");
                }
                QLangLine::Command(cmd) => {
                    if matches!(
                        cmd,
                        QLangCommand::Measure(_)
                            | QLangCommand::MeasureMany(_)
                            | QLangCommand::MeasureAll
                    ) {
                        self.collapsed = true;
                    }
                    self.push_ast(cmd.clone());
                }
            }
        }
        Ok(None)
    }

    pub fn run_qlang_from_file(&mut self, file_path: &str) {
        if let Err(error) = self.try_run_qlang_from_file(file_path) {
            eprintln!("{error}");
        }
    }

    pub fn try_run_qlang_from_file(&mut self, file_path: &str) -> Result<Option<Vec<u8>>, String> {
        let program = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read '{}': {}", file_path, e))?;
        self.append_from_lines(program.lines());
        self.run_parsed_commands()?;
        self.try_run()?;
        Ok(None)
    }

    pub fn append_from_str(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            return;
        }
        self.parser.append(line)
    }

    pub fn append_from_lines(&mut self, lines: Lines) {
        for line in lines {
            self.parser.append(line);
        }
    }

    pub fn reset(&mut self) {
        self.clear_ast();
        self.collapsed = false;
        self.parser = QLangParser::new();
        self.qvm.reset();
        self.variables.clear();
    }

    /// Discards the parsed program while preserving the current quantum state.
    ///
    /// This is useful for embedding APIs where each submitted source string is
    /// a new unit of execution, but gates should continue from the current QVM
    /// state.
    pub fn clear_program(&mut self) {
        self.ast.clear();
        self.parser = QLangParser::new();
        self.variables.clear();
        self.functions.clear();
        self.collapsed = false;
    }

    pub fn clear_ast(&mut self) {
        self.ast.clear();
    }

    fn push_ast(&mut self, cmd: QLangCommand) {
        self.ast.push(cmd.clone());
    }

    pub fn teardown(&mut self) {
        self.qvm.teardown();
    }
}
