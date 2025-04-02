use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::qvm::cuda::types::CudaComplex;

pub trait QuantumBackend {
    fn num_qubits(&self) -> usize;
    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize);
    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize);
    fn measure(&mut self, qubit: usize) -> u8;
    fn measure_all(&mut self) -> Vec<u8>;
    fn display(&self);
    fn reset(&mut self, num_qubits: usize);
    fn state_vector(&self) -> Vec<CudaComplex>;
}