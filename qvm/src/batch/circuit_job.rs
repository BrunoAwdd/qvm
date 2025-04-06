use crate::qlang::ast::QLangCommand;

#[derive(Clone)]
pub struct CircuitJob {
    pub num_qubits: usize,
    pub commands: Vec<QLangCommand>,
}
