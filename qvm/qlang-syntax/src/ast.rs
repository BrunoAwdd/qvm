use std::fmt;

// ── Operators ─────────────────────────────────────────────────────────────────

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
pub enum TypeAnnotation {
    Qubit,
    Bit,
    Int,
    Float,
    Bool,
    Array(Box<TypeAnnotation>),
    Void,
}

impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeAnnotation::Qubit => write!(f, "qubit"),
            TypeAnnotation::Bit => write!(f, "bit"),
            TypeAnnotation::Int => write!(f, "int"),
            TypeAnnotation::Float => write!(f, "float"),
            TypeAnnotation::Bool => write!(f, "bool"),
            TypeAnnotation::Array(inner) => write!(f, "[{}]", inner),
            TypeAnnotation::Void => write!(f, "void"),
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
    Array(Vec<Expression>),
    Index(Box<Expression>, Box<Expression>),
    Call(String, Vec<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n),
            Expression::Variable(v) => write!(f, "{}", v),
            Expression::BinaryOp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Expression::Measure(e) => write!(f, "measure({})", e),
            Expression::Array(es) => {
                write!(f, "[")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Expression::Index(a, i) => write!(f, "{}[{}]", a, i),
            Expression::Call(name, args) => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum QLangCommand {
    Create(usize),
    ApplyGate(String, Vec<Expression>),
    MeasureAll,
    Measure(usize),
    MeasureMany(Vec<usize>),
    Display,
    ListFunctions,
    Import {
        path: String,
    },
    FunctionDef {
        name: String,
        params: Vec<String>,
        param_types: Vec<Option<TypeAnnotation>>,
        return_type: Option<TypeAnnotation>,
        body: Vec<QLangCommand>,
    },
    Let {
        name: String,
        type_ann: Option<TypeAnnotation>,
        value: Expression,
    },
    Assign {
        name: String,
        value: Expression,
    },
    If {
        condition: Expression,
        then_branch: Vec<QLangCommand>,
        else_branch: Option<Vec<QLangCommand>>,
    },
    While {
        condition: Expression,
        body: Vec<QLangCommand>,
    },
    For {
        var: String,
        start: Expression,
        end: Expression,
        body: Vec<QLangCommand>,
    },
    QIf {
        condition: Expression,
        then_branch: Vec<QLangCommand>,
        else_branch: Option<Vec<QLangCommand>>,
    },
    Return(Expression),
    Break,
    Continue,
}

impl fmt::Display for QLangCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLangCommand::Create(n) => writeln!(f, "create({})", n),
            QLangCommand::ApplyGate(name, args) => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                writeln!(f, ")")
            }
            QLangCommand::MeasureAll => writeln!(f, "measure_all()"),
            QLangCommand::Measure(q) => writeln!(f, "measure({})", q),
            QLangCommand::MeasureMany(qs) => {
                writeln!(
                    f,
                    "measure({})",
                    qs.iter()
                        .map(|q| q.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            QLangCommand::Display => writeln!(f, "display()"),
            QLangCommand::ListFunctions => writeln!(f, "list_functions()"),
            QLangCommand::Import { path } => writeln!(f, "import \"{}\"", path),
            QLangCommand::FunctionDef {
                name,
                params,
                param_types,
                return_type,
                body,
            } => {
                write!(f, "fn {}(", name)?;
                for (i, (p, t)) in params.iter().zip(param_types.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                    if let Some(ty) = t {
                        write!(f, ": {}", ty)?;
                    }
                }
                write!(f, ")")?;
                if let Some(ret) = return_type {
                    write!(f, " -> {}", ret)?;
                }
                writeln!(f, " {{")?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
            QLangCommand::Let {
                name,
                type_ann,
                value,
            } => {
                write!(f, "let {}", name)?;
                if let Some(t) = type_ann {
                    write!(f, ": {}", t)?;
                }
                writeln!(f, " = {}", value)
            }
            QLangCommand::Assign { name, value } => writeln!(f, "{} = {}", name, value),
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
            QLangCommand::While { condition, body } => {
                writeln!(f, "while ({}) {{", condition)?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
            QLangCommand::For {
                var,
                start,
                end,
                body,
            } => {
                writeln!(f, "for ({} in {}..{}) {{", var, start, end)?;
                for cmd in body {
                    write!(f, "    {}", cmd)?;
                }
                writeln!(f, "}}")
            }
            QLangCommand::QIf {
                condition,
                then_branch,
                else_branch,
            } => {
                writeln!(f, "qif ({}) {{", condition)?;
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
            QLangCommand::Return(expr) => writeln!(f, "return {}", expr),
            QLangCommand::Break => writeln!(f, "break"),
            QLangCommand::Continue => writeln!(f, "continue"),
        }
    }
}

#[derive(Clone)]
pub struct AstController {
    ast: Vec<QLangCommand>,
}

impl AstController {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            ast: vec![QLangCommand::Create(num_qubits)],
        }
    }
    pub fn append(&mut self, cmd: &QLangCommand) {
        self.ast.push(cmd.clone());
    }
    pub fn to_source(&self) -> String {
        self.ast.iter().map(|c| c.to_string()).collect()
    }
    pub fn commands(&self) -> &[QLangCommand] {
        &self.ast
    }
    pub fn clear(&mut self) {
        self.ast.clear();
    }
}
