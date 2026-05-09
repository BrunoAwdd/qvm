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
        self.ast.iter().map(|cmd| cmd.to_string()).collect::<Vec<_>>().join("\n")
    }

    pub fn append(&mut self, cmd: QLangCommand) {
        if matches!(cmd, QLangCommand::MeasureAll) { self.collapsed = true; }
        self.ast.push(cmd);
    }

    /// Runs the type-checker first; aborts with typed errors if any are found.
    pub fn run(&mut self) {
        let mut tc = type_checker::TypeChecker::new(self.qvm.num_qubits());
        let errors = tc.check(&self.ast);
        if !errors.is_empty() {
            for e in &errors { eprintln!("[QLang type error] {}", e); }
            return;
        }
        let _ = run_ast(&mut self.qvm, &self.ast, &mut self.variables, &mut self.functions);
    }

    pub fn run_from_str(&mut self, code: &str) {
        self.append_from_lines(code.lines());
        let _ = self.run_parsed_commands();
        self.run();
    }

    pub fn run_parsed_commands(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.parser.validate_lines();
        let commands = self.parser.get_commands().to_vec();
        let mut final_result: Option<Vec<u8>> = None;

        for cmd in commands {
            match cmd {
                QLangLine::Command(QLangCommand::ApplyGate(ref name, ref args)) => {
                    self.push_ast(QLangCommand::ApplyGate(name.clone(), args.clone()));
                }
                QLangLine::Command(QLangCommand::Import { ref path }) => {
                    let file_path = if path == "std" { "std.ql".to_string() }
                    else if path.ends_with(".ql") { path.clone() }
                    else { format!("{}.ql", path) };
                    match fs::read_to_string(&file_path) {
                        Ok(content) => {
                            let mut parser = QLangParser::new();
                            for line in content.lines() {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() && !trimmed.starts_with("//") {
                                    parser.append(line);
                                }
                            }
                            parser.validate_lines();
                            for cmd in parser.get_commands() {
                                if let QLangLine::Command(QLangCommand::FunctionDef { name, params, param_types, return_type, body }) = cmd {
                                    self.functions.insert(name.clone(), QLangCommand::FunctionDef {
                                        name: name.clone(), params: params.clone(),
                                        param_types: param_types.clone(), return_type: return_type.clone(),
                                        body: body.clone(),
                                    });
                                }
                            }
                        }
                        Err(e) => { return Err(format!("Failed to import '{}': {}", path, e)); }
                    }
                }
                QLangLine::Command(QLangCommand::FunctionDef { ref name, ref params, ref param_types, ref return_type, ref body }) => {
                    self.functions.insert(name.clone(), QLangCommand::FunctionDef {
                        name: name.clone(), params: params.clone(),
                        param_types: param_types.clone(), return_type: return_type.clone(),
                        body: body.clone(),
                    });
                    self.push_ast(QLangCommand::FunctionDef {
                        name: name.clone(), params: params.clone(),
                        param_types: param_types.clone(), return_type: return_type.clone(),
                        body: body.clone(),
                    });
                }
                QLangLine::Command(QLangCommand::Create(n)) => {
                    if self.ast.iter().any(|cmd| matches!(cmd, QLangCommand::Create(_))) { continue; }
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
                QLangLine::Command(QLangCommand::Measure(q)) => {
                    self.push_ast(QLangCommand::Measure(q));
                    self.run();
                    let result = self.qvm.measure(q);
                    self.clear_ast();
                    final_result = Some(vec![result]);
                }
                QLangLine::Command(QLangCommand::MeasureAll) => {
                    self.append(QLangCommand::MeasureAll);
                    self.run();
                    let results = self.qvm.measure_all();
                    self.clear_ast();
                    final_result = Some(results);
                }
                QLangLine::Run => { self.run(); self.clear_ast(); }
                QLangLine::Reset => { self.reset(); println!("Reset"); }
                QLangLine::Command(cmd) => { self.push_ast(cmd.clone()); }
            }
        }
        Ok(final_result)
    }

    pub fn run_qlang_from_file(&mut self, file_path: &str) {
        let program = fs::read_to_string(file_path).expect("Erro ao ler o arquivo");
        self.append_from_lines(program.lines());
        let _ = self.run_parsed_commands();
    }

    pub fn append_from_str(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") { return; }
        self.parser.append(line)
    }

    pub fn append_from_lines(&mut self, lines: Lines) {
        for line in lines { self.parser.append(line); }
    }

    pub fn reset(&mut self) {
        self.clear_ast();
        self.collapsed = false;
        self.parser = QLangParser::new();
        self.qvm.reset();
        self.variables.clear();
    }

    pub fn clear_ast(&mut self) { self.ast.clear(); }

    fn push_ast(&mut self, cmd: QLangCommand) { self.ast.push(cmd.clone()); }

    pub fn teardown(&mut self) { self.qvm.teardown(); }
}