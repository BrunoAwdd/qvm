use crate::qlang::QLangCommand;

#[derive(Clone)]
pub struct CircuitJob {
    pub num_qubits: usize,
    pub commands: Vec<QLangCommand>,
}
