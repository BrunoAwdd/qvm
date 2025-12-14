pub mod backend;
pub mod cpu_backend;
pub mod cuda;
pub mod cuda_backend;
pub mod tensor_backend;
pub mod util;

/*
mod selected_backend {
    // Se 'cuda' estiver ativado (e for exclusivo), use CUDA
    #[cfg(all(feature = "cuda", not(any(feature = "tensor"))))]
    pub use crate::qvm::cuda_backend::CudaBackend as Backend;

    // Se 'tensor' estiver ativado (e não estiver 'cuda'), use Tensor
    #[cfg(all(feature = "tensor", not(feature = "cuda")))]
    pub use crate::qvm::tensor_backend::TensorBackend as Backend;

    // Se nenhuma feature estiver ativa, use CPU como fallback
    #[cfg(not(any(feature = "cuda", feature = "tensor")))]
    pub use crate::qvm::cpu_backend::CpuBackend as Backend;
}
use selected_backend::Backend;*/

use ndarray::Array2;

// traits
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::qvm::backend::QuantumBackend;
use crate::types::qlang_complex::QLangComplex;

#[derive(Clone)]
pub enum BackendType {
    Cpu(crate::qvm::cpu_backend::CpuBackend),
    Cuda(crate::qvm::cuda_backend::CudaBackend),
    Tensor(crate::qvm::tensor_backend::TensorBackend),
}

impl BackendType {
    pub fn from_compiled_features(num_qubits: usize) -> Self {
        match Self::feat() {
            "cpu" => Self::Cpu(crate::qvm::cpu_backend::CpuBackend::new(num_qubits)),
            "cuda" => Self::Cuda(crate::qvm::cuda_backend::CudaBackend::new(num_qubits)),
            "tensor" => Self::Tensor(crate::qvm::tensor_backend::TensorBackend::new(num_qubits)),
            _ => Self::Cpu(crate::qvm::cpu_backend::CpuBackend::new(num_qubits)),
        }
    }

    pub fn feat() -> &'static str {
        if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "tensor") {
            "tensor"
        } else if cfg!(feature = "cpu") {
            "cpu"
        } else {
            "cpu" // Todo return to cpu
        }
    }
}

impl QuantumBackend for BackendType {
    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        match self {
            BackendType::Cpu(b) => b.apply_gate(gate, qubit),
            BackendType::Cuda(b) => b.apply_gate(gate, qubit),
            BackendType::Tensor(b) => b.apply_gate(gate, qubit),
        }
    }

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        match self {
            BackendType::Cpu(b) => b.apply_gate_2q(gate, q0, q1),
            BackendType::Cuda(b) => b.apply_gate_2q(gate, q0, q1),
            BackendType::Tensor(b) => b.apply_gate_2q(gate, q0, q1),
        }
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        match self {
            BackendType::Cpu(b) => b.apply_gate_3q(gate, q0, q1, q2),
            BackendType::Cuda(b) => b.apply_gate_3q(gate, q0, q1, q2),
            BackendType::Tensor(b) => b.apply_gate_3q(gate, q0, q1, q2),
        }
    }

    fn measure(&mut self, qubit: usize) -> u8 {
        match self {
            BackendType::Cpu(b) => b.measure(qubit),
            BackendType::Cuda(b) => b.measure(qubit),
            BackendType::Tensor(b) => b.measure(qubit),
        }
    }

    fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        match self {
            BackendType::Cpu(b) => b.measure_many(qubits),
            BackendType::Cuda(b) => b.measure_many(qubits),
            BackendType::Tensor(b) => b.measure_many(qubits),
        }
    }

    fn measure_all(&mut self) -> Vec<u8> {
        match self {
            BackendType::Cpu(b) => b.measure_all(),
            BackendType::Cuda(b) => b.measure_all(),
            BackendType::Tensor(b) => b.measure_all(),
        }
    }

    fn display(&self) {
        match self {
            BackendType::Cpu(b) => b.display(),
            BackendType::Cuda(b) => b.display(),
            BackendType::Tensor(b) => b.display(),
        }
    }

    fn num_qubits(&self) -> usize {
        match self {
            BackendType::Cpu(b) => b.num_qubits(),
            BackendType::Cuda(b) => b.num_qubits(),
            BackendType::Tensor(b) => b.num_qubits(),
        }
    }

    fn state_vector(&self) -> Vec<QLangComplex> {
        match self {
            BackendType::Cpu(b) => b.state_vector(),
            BackendType::Cuda(b) => b.state_vector(),
            BackendType::Tensor(b) => b.state_vector(),
        }
    }

    fn reset(&mut self, num_qubits: usize) {
        match self {
            BackendType::Cpu(b) => b.reset(num_qubits),
            BackendType::Cuda(b) => b.reset(num_qubits),
            BackendType::Tensor(b) => b.reset(num_qubits),
        }
    }

    fn box_clone(&self) -> Box<dyn QuantumBackend> {
        match self {
            BackendType::Cpu(b) => Box::new(b.clone()),
            BackendType::Cuda(b) => Box::new(b.clone()),
            BackendType::Tensor(b) => Box::new(b.clone()),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            BackendType::Cpu(b) => b.name(),
            BackendType::Cuda(b) => b.name(),
            BackendType::Tensor(b) => b.name(),
        }
    }
}

impl Drop for QVM {
    fn drop(&mut self) {
        println!("🧹 Drop: resetando backend com teardown");
        //self.backend.reset(self.backend.num_qubits());
    }
}

pub struct QVM {
    pub backend: BackendType, // Backend do QVM
}

impl QVM {
    /// Inicializa o QVM com um dado número de qubits
    pub fn new(num_qubits: usize) -> Self {
        let backend: BackendType = BackendType::from_compiled_features(num_qubits);
        Self { backend }
    }

    /// Aplica uma porta quântica ao QVM
    pub fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        self.backend.apply_gate(gate, qubit);
    }

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        self.backend.apply_gate_2q(gate, q0, q1); // Aplica o gate ao backend
    }

    pub fn apply_gate_3q(
        &mut self,
        gate: &impl QuantumGateAbstract,
        q0: usize,
        q1: usize,
        q2: usize,
    ) {
        self.backend.apply_gate_3q(gate, q0, q1, q2);
    }

    pub fn measure(&mut self, qubit: usize) -> u8 {
        self.backend.measure(qubit) // Mede um qubit no backend
    }

    pub fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        self.backend.measure_many(qubits) // Mede vários qubits no backend
    }

    /// Mede o estado de todos os qubits
    pub fn measure_all(&mut self) -> Vec<u8> {
        self.backend.measure_all() // Mede todos os qubits no backend
    }

    /// Exibe o estado do QVM
    pub fn display(&self) {
        println!("Estado atual do QVM:");
        self.backend.display(); // Mostra o estado atual do backend
    }

    pub fn num_qubits(&self) -> usize { self.backend.num_qubits() }

    pub fn state_vector(&self) -> Vec<QLangComplex> { self.backend.state_vector() }

    pub fn teardown(&mut self) {
        self.backend = BackendType::from_compiled_features(self.backend.num_qubits());
    }

    pub fn reset(&mut self) { self.backend.reset(self.backend.num_qubits()); }

    pub fn box_clone(&self) -> Box<dyn QuantumBackend> { self.backend.box_clone() }

    pub fn estimate_entanglement(&self) -> f64 {
        let state = self.state_vector();
        let n = self.num_qubits();
        if n < 2 {
            return 0.0;
        }

        // Matriz densidade reduzida do primeiro qubit (2x2)
        let mut rho = ndarray::Array2::<QLangComplex>::zeros((2, 2));
        for i in 0..state.len() {
            for j in 0..state.len() {
                // Se os bits 1..N-1 forem iguais, some
                if (i & !1) == (j & !1) {
                    let a = state[i];
                    let b = state[j];
                    let idx_i = i & 1;
                    let idx_j = j & 1;
                    // a * b.conj()
                    let prod = QLangComplex {
                        re: a.re * b.re + a.im * b.im,
                        im: a.im * b.re - a.re * b.im,
                    };
                    rho[[idx_i, idx_j]] += prod;
                }
            }
        }

        // Converta para Complex64 para autovalores
        let rho_c64 = crate::types::qlang_complex::to_complex64(&rho);
        let (vals, _vecs) = ndarray_linalg::eig::Eig::eig(&rho_c64).unwrap();

        // Entropia de von Neumann: -Tr(rho log2 rho)
        let mut entropy = 0.0;
        for val in vals.iter() {
            let p = val.re;
            if p > 1e-12 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

impl Clone for QVM {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::{gates::one_q::{hadamard::Hadamard, pauli_x::PauliX}, qlang::{self, QLang}, qvm::QVM};

    #[test]
    fn test_qvm_basic() {
        let mut qvm = QVM::new(1);
        qvm.apply_gate(&PauliX::new(), 0);
        let result = qvm.measure(0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_qvm_clone() {
        let mut qvm = QVM::new(1);
        qvm.apply_gate(&PauliX::new(), 0);

        let clone = qvm.clone();
        assert_eq!(clone.num_qubits(), 1);
    }

    #[test]
    fn test_entanglement_hadamard() {
        let mut qlang = QLang::new(2);
        qlang.append_from_str("h(0)");
        qlang.run_parsed_commands();
        println!("RUNNING QLang with Hadamard gate on qubit 0");
        qlang.run();
        qlang.qvm.display();

        // Estado esperado: (|00⟩ + |10⟩)/√2, que não é emaranhado (entropia ~0)
        let ent = qlang.qvm.estimate_entanglement();
        println!("Entanglement (Hadamard on qubit 0, 2 qubits): {}", ent);
        assert!(ent < 0.01, "Entanglement should be near zero for separable state");

        // Agora cria um estado Bell (máximo emaranhamento)
        qlang.reset();

        qlang.append_from_str("h(0)");
        qlang.append_from_str("cnot(0, 1)");
        qlang.run_parsed_commands();
        qlang.run();
        qlang.qvm.display();


        let ent_bell = qlang.qvm.estimate_entanglement();
        println!("Entanglement (Bell state): {}", ent_bell);
        assert!(ent_bell > 0.9, "Entanglement should be high for Bell state");
    }
}
