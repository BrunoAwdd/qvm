use crate::qvm::QVM;
use crate::gates::{hadamard::Hadamard, pauli_x::PauliX, pauli_y::PauliY, pauli_z::PauliZ, cnot::CNOT};
use regex::Regex;
use std::fs;

pub enum QLangCommand {
    Create(usize),
    ApplyGate(String, Vec<usize>),
    MeasureAll,
    Display,
}

pub struct QLang {
    pub qvm: QVM,
    pub ast: Vec<QLangCommand>,
    pub collapsed: bool,
    func_regex: Regex,
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

    pub fn append(&mut self, cmd: QLangCommand) {
        if matches!(cmd, QLangCommand::MeasureAll) {
            self.collapsed = true;
        }
        self.ast.push(cmd);
    }

    pub fn run(&mut self) {
        for cmd in &self.ast {
            match cmd {
                QLangCommand::Create(n) => {
                    self.qvm = QVM::new(*n);
                }
                QLangCommand::ApplyGate(name, args) => {
                    match name.as_str() {
                        "hadamard" | "h" => {
                            let g = Hadamard::new();
                            self.qvm.apply_gate(&g, args[0]);
                        }
                        "paulix" | "x" => {
                            let g = PauliX::new();
                            self.qvm.apply_gate(&g, args[0]);
                        }
                        "pauliy" | "y" => {
                            let g = PauliY::new();
                            self.qvm.apply_gate(&g, args[0]);
                        }
                        "pauliz" | "z" => {
                            let g = PauliZ::new();
                            self.qvm.apply_gate(&g, args[0]);
                        }
                        "cnot" | "cx" => {
                            let g = CNOT::new();
                            self.qvm.apply_gate_2q(&g, args[0], args[1]);
                        }
                        _ => println!("Gate desconhecido: {}", name),
                    }
                }
                QLangCommand::Display => {
                    self.qvm.display();
                }
                QLangCommand::MeasureAll => {
                    self.qvm.measure_all();
                }
            }
        }
    }

    pub fn run_from_str(&mut self, code: &str) {
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            self.run_qlang_from_line(trimmed);
        }

        self.run(); 
    }

    pub fn run_qlang_from_line(&mut self, line: &str) {
        if let Some(caps) = self.func_regex.captures(line.trim()) {
            let raw = caps.get(1).unwrap().as_str();
            let args_str = caps.get(2).unwrap().as_str();
            let args: Vec<usize> = args_str
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().expect("Argumento inválido"))
                .collect();

            let canonical_name = match raw {
                "h" => "hadamard",
                "x" => "paulix",
                "y" => "pauliy",
                "z" => "pauliz",
                "cx" => "cnot",
                "m" => "measure_all",
                "d" => "display",
                other => other,
            };

            match canonical_name {
                "create" => {
                    self.ast.push(QLangCommand::Create(args[0]));
                }
                "hadamard" | "paulix" | "pauliy" | "pauliz" => {
                    self.ast.push(QLangCommand::ApplyGate(canonical_name.to_string(), vec![args[0]]));
                }
                "cnot" => {
                    self.ast.push(QLangCommand::ApplyGate("cnot".to_string(), vec![args[0], args[1]]));
                }
                "measure_all" => {
                    self.ast.push(QLangCommand::MeasureAll);
                }
                "display" => {
                    self.ast.push(QLangCommand::Display);
                }
                _ => println!("Comando desconhecido: {}", canonical_name),
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
            self.run_qlang_from_line(trimmed);
        }
    }

    pub fn reset(&mut self) {
        let qubits = self.qvm.state.num_qubits;
        self.qvm = QVM::new(qubits);
        self.ast.clear();
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
}
