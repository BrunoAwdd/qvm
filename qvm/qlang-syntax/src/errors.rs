use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum QLangError {
    #[error("unknown variable '{0}'")]
    UnknownVariable(String),

    #[error("unknown gate or function '{0}'")]
    UnknownGate(String),

    #[error("'{name}' expects {expected} arguments, got {got}")]
    ArityMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    #[error("index {index} out of bounds (array length {len})")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("cannot index a non-array value")]
    IndexOnNonArray,

    #[error("qubit {0} has already been measured (linear type violation)")]
    QubitAlreadyMeasured(usize),

    #[error("qubit index {0} is out of range for this register")]
    QubitOutOfRange(usize),

    #[error("only quantum gate calls are allowed inside qif blocks; measurements collapse state")]
    NonGateInQif,

    #[error("type error: {0}")]
    TypeError(String),

    #[error("import failed: {0}")]
    ImportError(String),
}

#[derive(Error, Debug)]
pub enum QLangParserError {
    #[error("Invalid Regex: {0}")]
    InvalidRegex(String),
    #[error("Invalid function name: {0}")]
    InvalidFunctionName(String),
    #[error("Invalid argument list: {0}")]
    InvalidArgumentList(String),
    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),
    #[error("Unexpected end of input")]
    UnexpectedEof,
}
