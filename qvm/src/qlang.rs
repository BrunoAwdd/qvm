pub mod aliases;
pub mod apply;
pub mod ast;
pub mod parser;
pub mod interpreter;
pub mod validators;

use regex::Regex;
use std::fs;

use crate::qlang::{ast::QLangCommand, parser::{QLangParser, QLangLine}, interpreter::run_ast, validators::validate_gate_arity};
use crate::qvm::{QVM, backend::QuantumBackend};

pub struct QLang {
    pub qvm: QVM,
    pub ast: Vec<QLangCommand>,
    pub collapsed: bool,
    func_regex: Regex,
}

impl Clone for QLang {
    fn clone(&self) -> Self {
        Self {
            qvm: self.qvm.clone(),
            ast: self.ast.clone(),
            collapsed: self.collapsed,
            func_regex: Regex::new(self.func_regex.as_str()).unwrap(),
        }
    }
}

impl QLang {
    pub fn new(num_qubits: usize) -> Self {
        let qvm = QVM::new(num_qubits);
        let func_regex = Regex::new(r"(\w+)\((.*)\)").unwrap();
        Self {
            qvm,
            ast: vec![QLangCommand::Create(num_qubits)],
            collapsed: false,
            func_regex,
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
        self.push_ast(cmd);
    }

    pub fn run(&mut self) {
        run_ast(&mut self.qvm, &self.ast);
    }

    pub fn run_from_str(&mut self, code: &str) {
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            let _ = self.run_qlang_from_line(trimmed);
        }

        self.run(); 
    }

    pub fn run_qlang_from_line(&mut self, line: &str) -> Result<Option<Vec<u8>>, String>  {
        let parser = QLangParser::new();
        match parser.parse_line(line) {
            Ok(QLangLine::Command(QLangCommand::ApplyGate(ref name, ref args))) => {
                let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                validate_gate_arity(name, self.qvm.num_qubits(), &args_refs)?;
                self.push_ast(QLangCommand::ApplyGate(name.clone(), args.clone()));
                Ok(None)
            }
            Ok(QLangLine::Command(QLangCommand::Create(n))) => {
                if self.ast.iter().any(|cmd| matches!(cmd, QLangCommand::Create(_))) {
                    return Ok(None);
                }
                self.reset();
                self.push_ast(QLangCommand::Create(n));
                Ok(None)
            }
            Ok(QLangLine::Command(QLangCommand::MeasureMany(qs))) => {
                println!("MeasureMany");
                self.run();
                let results = qs.iter().map(|&q| self.qvm.measure(q)).collect();
                self.clear_ast();

                Ok(Some(results))
            }
            Ok(QLangLine::Command(QLangCommand::MeasureAll)) => {
                self.run();
                let results = self.qvm.measure_all();
                self.clear_ast();
                Ok(Some(results))
            }
            Ok(QLangLine::Run) => {
                self.run();
                self.clear_ast();
                Ok(None)
            }
            Ok(QLangLine::Reset) => {
                self.reset();
                println!("Reset");
                Ok(None)
            }
            Ok(QLangLine::Command(cmd)) => {
                self.push_ast(cmd);
                Ok(None)
            }
            Err(e) => {
                eprintln!("Erro ao interpretar linha '{}': {}", line, e);
                Err(e)
            }
        }
    }

    pub fn run_qlang_from_file(&mut self, file_path: &str) {
        let program = fs::read_to_string(file_path)
            .expect("Erro ao ler o arquivo");

        for line in program.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            let _ = self.run_qlang_from_line(trimmed);
        }
    }

    pub fn reset(&mut self) {
        let qubits = self.qvm.backend.num_qubits();
        self.qvm = QVM::new(qubits);
        self.clear_ast();
        self.collapsed = false;
    }

    pub fn with_qvm(qvm: QVM) -> Self {
        let func_regex = Regex::new(r"(\w+)\((.*)\)").unwrap();
        Self {
            qvm,
            ast: vec![],
            collapsed: false,
            func_regex,
        }
    }

    fn push_ast(&mut self, cmd: QLangCommand) {
        println!("pushing ast: {:?}", cmd);
        self.ast.push(cmd);
    }

    pub fn clear_ast(&mut self) {
        println!("cleaning ast");
        self.ast.clear();
    }

} 

