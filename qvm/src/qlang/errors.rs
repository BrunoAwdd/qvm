use thiserror::Error;

#[derive(Error, Debug)]
pub enum QLangError {

}

#[derive(Error, Debug)]
pub enum QLangParserError {
    #[error("Invalid Regex: {0}")]
    InvalidRegex(String),
    #[error("Invalid function name: {0}")]
    InvalidFunctionName(String),
    #[error("Invalid argument list: {0}")]
    InvalidArgumentList(String),
}