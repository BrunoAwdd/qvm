use thiserror::Error;

/// Represents possible errors that can occur while running a batch of quantum circuits.
#[derive(Debug, Error)]
pub enum BatchRunnerError {
    /// Returned when there is a failure reading a batch file or resource.
    ///
    /// This wraps standard `std::io::Error`.
    #[error("Error reading file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Validation failed in job {job_index}: {reason}")]
    ValidationError {
        job_index: usize,
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*; // Import the necessary components
    use std::io;

    #[test]
    fn test_io_error() {
        // Simulate an I/O error (like a file read failure)
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");

        // Wrap it into BatchRunnerError::IoError
        let batch_error = BatchRunnerError::IoError(io_error);

        // Assert that the error matches the expected type and message
        assert_eq!(format!("{}", batch_error), "Error reading file: File not found");
    }

    #[test]
    fn test_validation_error() {
        // Create a ValidationError with a specific job index and reason
        let job_index = 1;
        let reason = "Missing qubit specification".to_string();
        let validation_error = BatchRunnerError::ValidationError {
            job_index,
            reason: reason.clone(),
        };

        // Assert that the error message is correctly formatted
        assert_eq!(format!("{}", validation_error), "Validation failed in job 1: Missing qubit specification");

        // Ensure the job_index and reason are correctly set in the error
        if let BatchRunnerError::ValidationError { job_index: idx, reason: r } = validation_error {
            assert_eq!(idx, job_index);
            assert_eq!(r, reason);
        } else {
            panic!("Expected ValidationError variant");
        }
    }
}
