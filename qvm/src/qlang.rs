use crate::qvm::QVM;
use crate::gates::{
    hadamard::Hadamard, 
    pauli_x::PauliX, 
    pauli_y::PauliY, 
    pauli_z::PauliZ, 
    cnot::CNOT, 
    rx::RX, 
    ry::RY, 
    rz::RZ, 
    s::S,
    swap::Swap, 
    t::T,
    toffoli::Toffoli,
    fredkin::Fredkin,
    u3::U3
};
use regex::Regex;
use std::fs;

use crate::qvm::backend::QuantumBackend;


pub enum QLangCommand {
    Create(usize),
    ApplyGate(String, Vec<String>),
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
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "paulix" | "x" => {
                            let g = PauliX::new();
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "pauliy" | "y" => {
                            let g = PauliY::new();
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "pauliz" | "z" => {
                            let g = PauliZ::new();
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "rx" => {
                            let qubit = args[0].parse::<usize>().unwrap();
                            let theta = args[1].parse::<f64>().unwrap();
                            let g = RX::new(theta);
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "ry" => {
                            let qubit = args[0].parse::<usize>().unwrap();
                            let theta = args[1].parse::<f64>().unwrap();
                            let g = RY::new(theta);
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "rz" => {
                            let qubit = args[0].parse::<usize>().unwrap();
                            let theta = args[1].parse::<f64>().unwrap();
                            let g = RZ::new(theta);
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "s" => {
                            let g = S::new();
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "t" => {
                            let g = T::new();
                            let qubit = args[0].parse::<usize>().unwrap();
                            self.qvm.apply_gate(&g, qubit);
                        }
                        "cnot" | "cx" => {
                            let g = CNOT::new();
                            let q0 = args[0].parse::<usize>().unwrap();
                            let q1 = args[1].parse::<usize>().unwrap();
                            self.qvm.apply_gate_2q(&g, q0, q1);
                        }
                        "swap" => {
                            let g = Swap::new();
                            let q0 = args[0].parse::<usize>().unwrap();
                            let q1 = args[1].parse::<usize>().unwrap();
                            self.qvm.apply_gate_2q(&g, q0, q1);
                        }
                        "toffoli" => {
                            let g = Toffoli::new();
                            let c1 = args[0].parse::<usize>().unwrap();
                            let c2 = args[1].parse::<usize>().unwrap();
                            let target = args[2].parse::<usize>().unwrap();
                            self.qvm.apply_gate_3q(&g, c1, c2, target);
                        }
                        "fredkin" => {
                            let g = Fredkin::new();
                            let ctrl = args[0].parse::<usize>().unwrap();
                            let t1 = args[1].parse::<usize>().unwrap();
                            let t2 = args[2].parse::<usize>().unwrap();
                            self.qvm.apply_gate_3q(&g, ctrl, t1, t2);
                        }
                        "u3" => {
                            let qubit = args[0].parse::<usize>().unwrap();
                            let theta = args[1].parse::<f64>().unwrap();
                            let phi = args[2].parse::<f64>().unwrap();
                            let lambda = args[3].parse::<f64>().unwrap();
                            let g = U3::new(theta, phi, lambda);
                            self.qvm.apply_gate(&g, qubit);
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
            let args: Vec<String> = args_str
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().to_string())
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
                    let qubit = args[0].parse::<usize>().unwrap();
                    self.ast.push(QLangCommand::Create(qubit));
                }
                "hadamard" | "paulix" | "pauliy" | "pauliz" | "s" | "t" => {
                    self.ast.push(QLangCommand::ApplyGate(
                        canonical_name.to_string(), 
                        vec![args[0].clone()]
                    ));
                }
                "rx" | "ry" | "rz" => {
                    let q = args[0].parse::<usize>().unwrap();
                    let theta = args[1].parse::<f64>().unwrap();
                    self.ast.push(QLangCommand::ApplyGate(
                        canonical_name.to_string(),
                        vec![q.to_string(), theta.to_string()],
                    ));
                }
                "cnot" | "swap" => {
                    let q0 = args[0].parse::<usize>().unwrap();
                    let q1 = args[1].parse::<usize>().unwrap();

                    if q0 == q1 {
                        panic!("Os qubits usados no '{}' devem ser diferentes. Recebido: {}, {}", canonical_name, q0, q1);
                    }

                    self.ast.push(QLangCommand::ApplyGate(
                        canonical_name.to_string(),
                        vec![q0.to_string(), q1.to_string()],
                    ));
                }
                "toffoli" | "fredkin" => {
                    self.ast.push(QLangCommand::ApplyGate(
                        canonical_name.to_string(),
                        vec![args[0].clone(), args[1].clone(), args[2].clone()],
                    ));
                }
                "u3" => {
                    self.ast.push(QLangCommand::ApplyGate(
                        "u3".to_string(),
                        vec![
                            args[0].clone(), args[1].clone(), args[2].clone(), args[3].clone()
                        ],
                    ));
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
        let qubits = self.qvm.backend.num_qubits();
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
