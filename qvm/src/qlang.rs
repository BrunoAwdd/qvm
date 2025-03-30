use std::fs;
use regex::Regex;
use crate::qvm::QVM;
use crate::gates::{hadamard::Hadamard, pauli_x::PauliX, pauli_y::PauliY, pauli_z::PauliZ, cnot::CNOT};

pub struct QLang {
    pub qvm: QVM,  
    func_regex: Regex,
}

impl QLang {
    /// Inicializa o interpretador QLang com um número de qubits
    pub fn new(num_qubits: usize) -> Self {
        let qvm: QVM = QVM::new(num_qubits);
        let func_regex = Regex::new(r"(\w+)\((.*)\)").unwrap();
        Self { qvm, func_regex }
    }

    pub fn run_from_str(&mut self, code: &str) {
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            self.run_qlang_from_line(trimmed); // Ou o equivalente interno
        }
    }

    pub fn run_qlang_from_line(&mut self, line: &str) {
        if let Some(caps) = self.func_regex.captures(line.trim()) {
            let mut function_name = &caps[1];
            let args: Vec<&str> = caps[2].split(',').map(|s| s.trim()).collect();

            // Mapeamento de atalhos
            function_name = match function_name {
                "h" => "hadamard",
                "x" => "paulix",
                "y" => "pauliy",
                "z" => "pauliz",
                "cx" => "cnot",
                "m" => "measure_all",
                "d" => "display",
                _ => function_name,
            };

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
                        self.qvm.apply_gate_2q(&cnot_gate, control_qubit, target_qubit);
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


    /// Executa um arquivo QLang e aplica as operações quânticas no QVM
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

}
