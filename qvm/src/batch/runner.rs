
use crate::qlang::QLang;
use crate::qvm::QVM;
use super::{
    circuit_job::CircuitJob, 
    errors::BatchRunnerError
};
use rayon::prelude::*;

/// Executes multiple quantum circuit jobs in parallel.
///
/// `BatchRunner` is designed to run a list of pre-defined QLang jobs,
/// each represented by a `CircuitJob`. It supports both in-memory
/// construction and loading from `.qlang` source files.
///
/// This is useful for simulation benchmarking, batch testing,
/// or preparing circuits for backend dispatch.
pub struct BatchRunner {
    /// List of circuits to be run.
    pub jobs: Vec<CircuitJob>,
}

/// Creates a new `BatchRunner` from a list of circuit jobs.
///
/// # Parameters
/// - `jobs`: A list of `CircuitJob` instances to execute.
impl BatchRunner {
    pub fn new(jobs: Vec<CircuitJob>) -> Self {
        Self { jobs }
    }

    /// Runs all circuit jobs in parallel and returns their resulting `QVM`s.
    ///
    /// Each job is executed in its own isolated `QLang` interpreter.
    ///
    /// # Returns
    /// A `Vec<QVM>` representing the final state of each quantum virtual machine.
    ///
    /// # Note
    /// Requires the `rayon` crate for parallel execution
    pub fn run_all(&self) -> Vec<QVM> {
        self.jobs
            .par_iter()
            .map(|job| {
                let mut qlang = QLang::new(job.num_qubits);
                job.commands.iter().cloned().for_each(|cmd| qlang.append(cmd));
                qlang.run();
                qlang.qvm.clone()
            })
            .collect()
    }

    /// Loads circuit jobs from a list of `.qlang` source file paths.
    ///
    /// Each file is parsed and executed into a `CircuitJob` for batch processing.
    ///
    /// # Parameters
    /// - `paths`: A list of string paths to `.qlang` files.
    ///
    /// # Returns
    /// - `Ok(BatchRunner)` if all files are successfully parsed
    /// - `Err(BatchRunnerError)` if any file cannot be read
    ///
    /// # Errors
    /// - Returns `BatchRunnerError::IoError` for unreadable files
    ///
    /// # Example
    /// ```no_run
    /// let runner = BatchRunner::from_files(vec!["circuits/job1.qlang", "circuits/job2.qlang"])?;
    /// let results = runner.run_all();
    /// ```
    pub fn from_files(paths: Vec<&str>) -> Result<Self, BatchRunnerError> {
        let jobs: Result<Vec<CircuitJob>, BatchRunnerError> = paths.into_iter().map(|path| {
            let content = std::fs::read_to_string(path)
                .map_err(|e| BatchRunnerError::IoError(e))?;
            let mut qlang = QLang::new(1); 
            qlang.run_from_str(&content);
            Ok(CircuitJob {
                num_qubits: qlang.qvm.num_qubits(),
                commands: qlang.ast.clone(),
            })
        }).collect();

        let runner = Self { jobs: jobs? };
        Ok(runner)
    }


}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qlang::ast::QLangCommand;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_batch_runner_new() {
        let job = CircuitJob {
            num_qubits: 2,
            commands: vec![],
        };
        let runner = BatchRunner::new(vec![job.clone()]);
        assert_eq!(runner.jobs.len(), 1);
    }

    #[test]
    fn test_batch_runner_run_all() {
        let job = CircuitJob {
            num_qubits: 1,
            commands: vec![QLangCommand::ApplyGate("hadamard".to_string(), vec!["0".into()])],
        };

        let runner = BatchRunner::new(vec![job]);
        let results = runner.run_all();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_qubits(), 1);
        // Further assertions may depend on your QVM implementation.
    }

    #[test]
    fn test_batch_runner_from_files_success() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "h 0").unwrap();
        let path = temp_file.path().to_str().unwrap();

        let runner = BatchRunner::from_files(vec![path]).expect("Should parse file");
        assert_eq!(runner.jobs.len(), 1);
        assert_eq!(runner.jobs[0].num_qubits, 1);
    }

    #[test]
    fn test_batch_runner_from_files_failure() {
        let result = BatchRunner::from_files(vec!["nonexistent_file.qlang"]);
        assert!(matches!(result, Err(BatchRunnerError::IoError(_))));
    }
}
