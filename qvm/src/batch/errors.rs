use thiserror::Error;

#[derive(Debug, Error)]
pub enum BatchRunnerError {
    #[error("Error reading file: {0}")]
    IoError(#[from] std::io::Error),

    //#[error("Error interpreting file: {0}")]
    //InterpreterError(#[from] crate::qlang::parser::QLangParserError),

    //#[error("Error executing circuit: {0}")]
    //ExecutionError(#[from] crate::qvm::errors::QVMError),
}