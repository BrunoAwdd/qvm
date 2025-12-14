use std::fmt;

/// Represents a single quantum command in the QLang abstract syntax tree.
///
/// These commands form the building blocks of a quantum program.
/// They include quantum gate applications, measurement operations,
/// circuit setup (e.g., `create`), and utility commands like `display`.
#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Sub => write!(f, "-"),
            Operator::Mul => write!(f, "*"),
            Operator::Div => write!(f, "/"),
            Operator::Eq => write!(f, "=="),
            Operator::Neq => write!(f, "!="),
            Operator::Lt => write!(f, "<"),
            Operator::Gt => write!(f, ">"),
            Operator::Le => write!(f, "<="),
            Operator::Ge => write!(f, ">="),
            Operator::And => write!(f, "&&"),
            Operator::Or => write!(f, "||"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Variable(String),
    BinaryOp {
        left: Box<Expression>,
        op: Operator,
        right: Box<Expression>,
    },
    Measure(Box<Expression>),
    /// Array literal: `[1, 2, 3]`
    Array(Vec<Expression>),
    /// Array indexing: `arr[index]`
    Index(Box<Expression>, Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::BinaryOp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Expression::Measure(expr) => write!(f, "measure({})", expr),
            Expression::Array(exprs) => {
                write!(f, "[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", expr)?;
                }
                write!(f, "]")
            }
            Expression::Index(arr, idx) => write!(f, "{}[{}]", arr, idx),
        }
    }
}

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

    /// Applies a quantum gate or function.
    ///
    /// Syntax: `gate(arg1, arg2)`
    ApplyGate(String, Vec<Expression>),

    /// Measures all qubits in the register.
    ///
    /// Syntax: `measure_all()`
    MeasureAll,

    /// Measures a specific qubit.
    ///
    /// Syntax: `measure(q)`
    Measure(usize), // Deprecated? parser uses Expression::Measure now? 
                    // No, parser returns QLangCommand::Measure for standalone measure?
                    // Let's check parser. Parser uses QLangCommand::Measure(usize) for "measure(q)".
                    // But wait, Expression::Measure exists too.
                    // Let's keep Measure(usize) for now but maybe it should be Expression too?
                    // Actually, let's look at parser.
                    // Parser for "measure" command:
                    // if name == "measure" { ... QLangCommand::Measure(val as usize) ... }
                    // So it expects number literal.
                    // If we want measure(arr[0]), we need to update this too.
    
    /// Measures multiple qubits.
    ///
    /// Syntax: `measure(q1, q2)`
    MeasureMany(Vec<usize>),

    /// Displays the current quantum state (or equivalent debug info).
    ///
    /// Syntax: `display()`
    Display,

    /// Defines a new function/gate.
    ///
    /// Syntax: `fn name(p1, p2) { ... }`
    FunctionDef {
        name: String,
        params: Vec<String>,
        body: Vec<QLangCommand>,
    },

    /// Imports functions from another file.
    ///
    /// Syntax: `import "filename"`
    Import {
        path: String,
    },

    /// Lists all available functions.
    ///
    /// Syntax: `list_functions()`
    ListFunctions,

    /// Conditional execution.
    ///
    /// Syntax: `if (condition) { ... } else { ... }`
    If {
        condition: Expression,
        then_branch: Vec<QLangCommand>,
        else_branch: Option<Vec<QLangCommand>>,
    },

    /// Loop: While
    ///
    /// Syntax: `while (condition) { ... }`
    While {
        condition: Expression,
        body: Vec<QLangCommand>,
    },

    /// Loop: For
    ///
    /// Syntax: `for (var in start..end) { ... }`
    For {
        var: String,
        start: Expression,
        end: Expression,
        body: Vec<QLangCommand>,
    },

    /// Variable declaration.
    ///
    /// Syntax: `let name = value`
    Let { name: String, value: Expression },

    /// Variable assignment.
    ///
    /// Syntax: `name = value`
    Assign { name: String, value: Expression },
}

impl fmt::Display for QLangCommand {
    /// Formats the command as QLang source code, including a trailing newline.
    ///
    /// This is useful for pretty-printing the program or serializing it back to text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangCommand::Create(n) => writeln!(f, "create({})", n),
            QLangCommand::ApplyGate(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                writeln!(f, ")")
            }
            QLangCommand::MeasureAll => writeln!(f, "measure_all()"),
            QLangCommand::Measure(q) => writeln!(f, "measure({})", q),
            QLangCommand::MeasureMany(qs) => {
                let list = qs
                    .iter()
                    .map(|q| q.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(f, "measure({})", list)
            }
            QLangCommand::Display => writeln!(f, "display()"),
            QLangCommand::If {
                condition,
                then_branch,
                else_branch,
            } => {
                writeln!(f, "if ({}) {{", condition)?;
                for cmd in then_branch {
                    write!(f, "    {}", cmd)?;
                }
                write!(f, "}}")?;
                if let Some(else_cmds) = else_branch {
                    writeln!(f, " else {{")?;
                    for cmd in else_cmds {
                        write!(f, "    {}", cmd)?;
                    }
                    write!(f, "}}")?;
                }
                writeln!(f)
            }
            QLangCommand::Let { name, value } => writeln!(f, "let {} = {}", name, value),
            QLangCommand::Assign { name, value } => writeln!(f, "{} = {}", name, value),
            QLangCommand::FunctionDef { name, params, body } => {
                writeln!(f, "fn {}({}) {{", name, params.join(", "))?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
            QLangCommand::Import { path } => writeln!(f, "import \"{}\"", path),
            QLangCommand::ListFunctions => writeln!(f, "list_functions()"),
            QLangCommand::While { condition, body } => {
                writeln!(f, "while ({}) {{", condition)?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
            QLangCommand::For { var, start, end, body } => {
                writeln!(f, "for ({} in {}..{}) {{", var, start, end)?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
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

    pub fn append(&mut self, cmd: &QLangCommand) { self.ast.push(cmd.clone()); }

    pub fn to_source(&self) -> String { self.ast.iter().map(|cmd| cmd.to_string()).collect() }

    pub fn commands(&self) -> &[QLangCommand] { &self.ast }

    pub fn clear(&mut self) { self.ast.clear(); }
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
