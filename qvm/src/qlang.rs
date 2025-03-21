use crate::qvm::QVM;  // Importando a QVM
use std::fs;
use regex::Regex;
use crate::gates::{hadamard::Hadamard, pauli_x::PauliX, pauli_y::PauliY, pauli_z::PauliZ, cnot::CNOT};

pub struct QLang {
    pub qvm: QVM,  // A QVM será chamada pela QLang
}

impl QLang {
    /// Inicializa o interpretador QLang com um número de qubits
    pub fn new(num_qubits: usize) -> Self {
        let qvm: QVM = QVM::new(num_qubits);
        Self { qvm }
    }

    /// Executa um arquivo QLang e aplica as operações quânticas no QVM
    pub fn run_qlang_from_file(&mut self, file_path: &str) {
        let program: String = fs::read_to_string(file_path)
            .expect("Erro ao ler o arquivo");

        let lines: std::str::Lines<'_> = program.lines();
        let func_regex: Regex = Regex::new(r"(\w+)\((.*)\)").unwrap(); // Regex para funções com parênteses

        for line in lines {
            if let Some(caps) = func_regex.captures(line.trim()) {
                let function_name: &str = &caps[1];  // Nome da função (como 'create', 'hadamard', etc.)
                let args: Vec<&str> = caps[2].split(',').map(|s| s.trim()).collect(); // Argumentos da função

                match function_name {
                    "create" => {
                        let num_qubits: usize = args[0].parse().unwrap();
                        self.qvm.state = crate::state::quantum_state::QuantumState::new(num_qubits);
                    }
                    "hadamard" => {
                        let qubit: usize = args[0].parse().unwrap();
                        let h_gate: Hadamard = Hadamard::new();
                        self.qvm.apply_gate(&h_gate, qubit);  // Aplica o Hadamard ao qubit
                    }
                    "paulix" => {
                        let qubit: usize = args[0].parse().unwrap();
                        let p_x_gate: PauliX = PauliX::new();
                        self.qvm.apply_gate(&p_x_gate, qubit);  // Aplica o PauliX ao qubit
                    }
                    "pauliy" => {
                        let qubit: usize = args[0].parse().unwrap();
                        let p_y_gate: PauliY = PauliY::new();
                        self.qvm.apply_gate(&p_y_gate, qubit);  // Aplica o PauliY ao qubit
                    }
                    "pauliz" => {
                        let qubit: usize = args[0].parse().unwrap();
                        let p_z_gate: PauliZ = PauliZ::new();
                        self.qvm.apply_gate(&p_z_gate, qubit);  // Aplica o PauliZ ao qubit
                    }
                    "cnot" => {
                        let control_qubit: usize = args[0].parse().unwrap();
                        let target_qubit: usize = args[1].parse().unwrap();
                        let cnot_gate: CNOT = CNOT::new();
                        self.qvm.apply_gate(&cnot_gate, control_qubit);  // Aplica o CNOT ao qubit de controle
                        self.qvm.apply_gate(&cnot_gate, target_qubit);   // Aplica o CNOT ao qubit alvo
                    }
                    "measure_all" => {
                        let result = self.qvm.measure_all();
                        println!("Resultado da medição: {:?}", result);
                    }
                    "display" => {
                        self.qvm.display();
                    }
                    _ => println!("Comando desconhecido: {}", function_name),
                }
            }
        }
    }
}
