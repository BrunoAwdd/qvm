use std::fmt;

use super::parser::QLangLine;

/// Represents a single quantum command in the QLang abstract syntax tree.
///
/// These commands form the building blocks of a quantum program.
/// They include quantum gate applications, measurement operations,
/// circuit setup (e.g., `create`), and utility commands like `display`.
#[derive(Clone, Debug)]
pub enum QLangCommand {
    /// Creates a quantum register with the given number of qubits.
    ///
    /// Syntax: `create(n)`
    Create(usize),

    /// Applies a quantum gate with arguments (e.g., qubit indices or parameters).
    ///
    /// Syntax: `gate_name(arg1, arg2, ...)`
    ApplyGate(String, Vec<String>),

    /// Measures all qubits in the register.
    ///
    /// Syntax: `measure_all()`
    MeasureAll,

    /// Measures a single qubit.
    ///
    /// Syntax: `measure(q)`
    Measure(usize),

    /// Measures multiple qubits.
    ///
    /// Syntax: `measure(q1, q2, ...)`
    MeasureMany(Vec<usize>),

    /// Displays the current quantum state (or equivalent debug info).
    ///
    /// Syntax: `display()`
    Display,
}

impl fmt::Display for QLangCommand {
    /// Formats the command as QLang source code, including a trailing newline.
    ///
    /// This is useful for pretty-printing the program or serializing it back to text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangCommand::Create(n) => writeln!(f, "create({})", n),
            QLangCommand::ApplyGate(name, args) => {
                writeln!(f, "{}({})", name, args.join(","))
            }
            QLangCommand::MeasureAll => writeln!(f, "measure_all()"),
            QLangCommand::Measure(q) => writeln!(f, "measure({})", q),
            QLangCommand::MeasureMany(qs) => {
                let list = qs.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(",");
                writeln!(f, "measure({})", list)
            }
            QLangCommand::Display => writeln!(f, "display()"),
        }
    }
}

#[derive(Clone)]
pub struct AstController {
    ast: Vec<QLangCommand>,
}

impl AstController {
    pub fn new(num_qubits: usize) -> Self { 
        let init = vec![QLangCommand::Create(num_qubits)];
        Self { ast: init }    
    }

    pub fn append(&mut self, cmd: &QLangCommand) {
        self.ast.push(cmd.clone());
    }

    pub fn to_source(&self) -> String {
        self.ast.iter().map(|cmd| cmd.to_string()).collect()
    }

    pub fn commands(&self) -> &[QLangCommand] {
        &self.ast
    }

    pub fn clear(&mut self) {
        self.ast.clear();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_create() {
        let cmd = QLangCommand::Create(3);
        assert_eq!(cmd.to_string(), "create(3)\n");
    }

    #[test]
    fn display_apply_gate() {
        let cmd = QLangCommand::ApplyGate("h".into(), vec!["0".into()]);
        assert_eq!(cmd.to_string(), "h(0)\n");

        let cmd = QLangCommand::ApplyGate("rx".into(), vec!["0".into(), "3.14".into()]);
        assert_eq!(cmd.to_string(), "rx(0,3.14)\n");
    }

    #[test]
    fn display_measure_all() {
        let cmd = QLangCommand::MeasureAll;
        assert_eq!(cmd.to_string(), "measure_all()\n");
    }

    #[test]
    fn display_measure_single() {
        let cmd = QLangCommand::Measure(2);
        assert_eq!(cmd.to_string(), "measure(2)\n");
    }

    #[test]
    fn display_measure_many() {
        let cmd = QLangCommand::MeasureMany(vec![0, 1, 3]);
        assert_eq!(cmd.to_string(), "measure(0,1,3)\n");
    }

    #[test]
    fn display_display() {
        let cmd = QLangCommand::Display;
        assert_eq!(cmd.to_string(), "display()\n");
    }

    #[test]
    fn test_ast_controller_collects_commands() {
        let qubit = 2;

        let mut controller = AstController::new(qubit);

        let cmd = QLangCommand::Create(qubit);
        controller.append(&cmd);

        let source = controller.to_source();
        assert!(source.contains("create(2)"));
    }


}