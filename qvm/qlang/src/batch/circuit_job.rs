use crate::ast::QLangCommand;

/// Represents a batchable QLang circuit job.
///
/// A `CircuitJob` encapsulates a quantum circuit definition,
/// including the number of qubits and the full list of commands
/// (`QLangCommand`) to be executed.
///
/// It is designed to support offline processing, simulation,
/// remote execution, or parallel jobs submitted in bulk.
///
/// # Fields
/// - `num_qubits`: The number of qubits required for the circuit.
/// - `commands`: A list of commands that form the quantum program.
///
/// # Example
/// ```
/// use qlang::batch::circuit_job::CircuitJob;
/// use qlang::ast::{QLangCommand, Expression};
///
/// let job = CircuitJob {
///     num_qubits: 2,
///     commands: vec![
///         QLangCommand::Create(2),
///         QLangCommand::ApplyGate("h".into(), vec![Expression::Number(0.0)]),
///         QLangCommand::ApplyGate("cx".into(), vec![Expression::Number(0.0), Expression::Number(1.0)]),
///         QLangCommand::MeasureAll,
///     ],
/// };
/// ```

#[derive(Clone)]
pub struct CircuitJob {
    pub num_qubits: usize,
    pub commands: Vec<QLangCommand>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{QLangCommand, Expression};

    #[test]
    fn test_create_circuit_job() {
        // Creating a simple CircuitJob
        let job = CircuitJob {
            num_qubits: 2,
            commands: vec![
                QLangCommand::Create(2),
                QLangCommand::ApplyGate("h".into(), vec![Expression::Number(0.0)]),
                QLangCommand::ApplyGate("cx".into(), vec![Expression::Number(0.0), Expression::Number(1.0)]),
                QLangCommand::MeasureAll,
            ],
        };

        // Check if the number of qubits is correct
        assert_eq!(job.num_qubits, 2);

        // Check if the number of commands is correct
        assert_eq!(job.commands.len(), 4);

        // Check the type and order of commands
        assert!(matches!(job.commands[0], QLangCommand::Create(2)));
        assert!(
            matches!(job.commands[1], QLangCommand::ApplyGate(ref gate, ref qubits) if gate == "h" && qubits == &vec![Expression::Number(0.0)])
        );
        assert!(
            matches!(job.commands[2], QLangCommand::ApplyGate(ref gate, ref qubits) if gate == "cx" && qubits == &vec![Expression::Number(0.0), Expression::Number(1.0)])
        );
        assert!(matches!(job.commands[3], QLangCommand::MeasureAll));
    }

    #[test]
    fn test_empty_circuit_job() {
        // Creating an empty CircuitJob
        let job = CircuitJob {
            num_qubits: 0,
            commands: vec![],
        };

        // Check if the number of qubits is zero
        assert_eq!(job.num_qubits, 0);

        // Check if the list of commands is empty
        assert_eq!(job.commands.len(), 0);
    }

    #[test]
    fn test_circuit_job_with_different_qubits() {
        // Creating a CircuitJob with 3 qubits
        let job = CircuitJob {
            num_qubits: 3,
            commands: vec![
                QLangCommand::Create(3),
                QLangCommand::ApplyGate("h".into(), vec![Expression::Number(0.0)]),
                QLangCommand::ApplyGate("cx".into(), vec![Expression::Number(0.0), Expression::Number(2.0)]),
                QLangCommand::MeasureAll,
            ],
        };

        // Check if the number of qubits is correct
        assert_eq!(job.num_qubits, 3);

        // Check if the number of commands is correct
        assert_eq!(job.commands.len(), 4);

        // Check the integrity of the commands
        assert!(matches!(job.commands[0], QLangCommand::Create(3)));
        assert!(
            matches!(job.commands[1], QLangCommand::ApplyGate(ref gate, ref qubits) if gate == "h" && qubits == &vec![Expression::Number(0.0)])
        );
        assert!(
            matches!(job.commands[2], QLangCommand::ApplyGate(ref gate, ref qubits) if gate == "cx" && qubits == &vec![Expression::Number(0.0), Expression::Number(2.0)])
        );
        assert!(matches!(job.commands[3], QLangCommand::MeasureAll));
    }

    // Additional test for invalid commands or unexpected behaviors
    #[test]
    fn test_invalid_command_in_circuit_job() {
        // Creating a CircuitJob with an invalid gate command
        let job = CircuitJob {
            num_qubits: 2,
            commands: vec![QLangCommand::ApplyGate(
                "invalid_gate".into(),
                vec![Expression::Number(0.0)],
            )],
        };

        // Check if the invalid command was added
        assert!(
            matches!(job.commands[0], QLangCommand::ApplyGate(ref gate, _) if gate == "invalid_gate")
        );
    }
}
