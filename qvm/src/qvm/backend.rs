use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::types::qlang_complex::QLangComplex;

pub trait QuantumBackend {
    fn num_qubits(&self) -> usize;
    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize);
    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize);
    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize);
    fn measure(&mut self, qubit: usize) -> u8;
    fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8>;
    fn measure_all(&mut self) -> Vec<u8>;
    fn display(&self);
    fn reset(&mut self, num_qubits: usize);
    fn state_vector(&self) -> Vec<QLangComplex>;

    /// Clonagem dinâmica
    fn box_clone(&self) -> Box<dyn QuantumBackend>;
    fn name(&self) -> &'static str;
}

impl Clone for Box<dyn QuantumBackend> {
    fn clone(&self) -> Box<dyn QuantumBackend> {
        self.box_clone()
    }
}
