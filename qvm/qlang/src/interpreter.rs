use std::collections::HashMap;
use std::fmt;

use qlang_core::gates::{
    one_q::{
        hadamard::*, identity::*, pauli_x::*, pauli_y::*, pauli_z::*, s::*, s_dagger::*, t::*,
        t_dagger::*,
    },
    rotation_q::{phase::*, rx::*, ry::*, rz::*, u1::*, u2::*, u3::*},
    three_q::{fredkin::*, toffoli::*},
    two_q::{cnot::*, cy::*, cz::*, iswap::*, swap::*},
};
use crate::{apply::*, ast::{Expression, Operator, QLangCommand}};
use qvm::QVM;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Array(Vec<Value>),
    Qubit(usize),
    Bit(u8),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Qubit(i)  => write!(f, "qubit({})", i),
            Value::Bit(b)    => write!(f, "{}", b),
            Value::Array(a)  => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl Value {
    pub fn as_number(&self) -> f64 {
        match self {
            Value::Number(n) => *n,
            Value::Bit(b)    => *b as f64,
            Value::Qubit(_)  => panic!("Type error: qubit cannot be used as a number"),
            Value::Array(_)  => panic!("Type error: array cannot be used as a number"),
        }
    }

    pub fn as_qubit_index(&self) -> usize {
        match self {
            Value::Qubit(i)  => *i,
            Value::Number(n) => *n as usize,
            Value::Bit(b)    => *b as usize,
            Value::Array(_)  => panic!("Type error: array cannot be used as qubit index"),
        }
    }

    pub fn as_usize(&self) -> usize { self.as_qubit_index() }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Number(n) => *n != 0.0,
            Value::Bit(b)    => *b != 0,
            Value::Qubit(_)  => panic!("Type error: qubit cannot be used as a boolean"),
            Value::Array(a)  => !a.is_empty(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ControlFlow {
    None,
    Return(Value),
    Break,
    Continue,
}

pub fn run_ast(
    qvm: &mut QVM,
    ast: &[QLangCommand],
    variables: &mut HashMap<String, Value>,
    functions: &mut HashMap<String, QLangCommand>,
) -> ControlFlow {
    for cmd in ast {
        match execute_command(qvm, cmd, variables, functions) {
            ControlFlow::None => {}
            cf => return cf,
        }
    }
    ControlFlow::None
}

fn execute_command(
    qvm: &mut QVM,
    cmd: &QLangCommand,
    variables: &mut HashMap<String, Value>,
    functions: &mut HashMap<String, QLangCommand>,
) -> ControlFlow {
    match cmd {
        QLangCommand::Create(n) => {
            *qvm = QVM::new(*n);
            variables.clear();
        }

        QLangCommand::Import { .. } => {}

        QLangCommand::FunctionDef { name, .. } => {
            functions.insert(name.clone(), cmd.clone());
        }

        QLangCommand::ApplyGate(name, args) => {
            if let Some(QLangCommand::FunctionDef { params, body, .. }) =
                functions.get(name).cloned()
            {
                if args.len() != params.len() {
                    eprintln!("Error: '{}' expects {} arguments, got {}", name, params.len(), args.len());
                    return ControlFlow::None;
                }
                let mut local_vars = HashMap::new();
                for (param, arg_expr) in params.iter().zip(args.iter()) {
                    let val = evaluate_expression(arg_expr, variables, qvm, functions);
                    local_vars.insert(param.clone(), val);
                }
                match run_ast(qvm, &body, &mut local_vars, functions) {
                    ControlFlow::Break | ControlFlow::Continue => {
                        eprintln!("Warning: break/continue cannot escape a function body");
                    }
                    _ => {}
                }
            } else {
                let resolved = resolve_args(name, args, variables, qvm, functions);
                if let Some(args) = resolved {
                    apply_gate_dispatch(qvm, name, &args);
                }
            }
        }

        QLangCommand::Display    => qvm.display(),
        QLangCommand::MeasureAll => { qvm.measure_all(); }
        QLangCommand::Measure(q) => { qvm.measure(*q); }
        QLangCommand::MeasureMany(qs) => { qvm.measure_many(qs); }

        QLangCommand::Let { name, value, .. } => {
            let val = evaluate_expression(value, variables, qvm, functions);
            variables.insert(name.clone(), val);
        }

        QLangCommand::Assign { name, value } => {
            let val = evaluate_expression(value, variables, qvm, functions);
            variables.insert(name.clone(), val);
        }

        QLangCommand::If { condition, then_branch, else_branch } => {
            let cond = evaluate_expression(condition, variables, qvm, functions);
            if cond.is_truthy() {
                let cf = run_ast(qvm, then_branch, variables, functions);
                if !matches!(cf, ControlFlow::None) { return cf; }
            } else if let Some(else_cmds) = else_branch {
                let cf = run_ast(qvm, else_cmds, variables, functions);
                if !matches!(cf, ControlFlow::None) { return cf; }
            }
        }

        QLangCommand::QIf { condition, then_branch, else_branch } => {
            let control = evaluate_expression(condition, variables, qvm, functions).as_qubit_index();
            for cmd in then_branch {
                if let QLangCommand::ApplyGate(gate, args) = cmd {
                    let mut new_args = vec![control.to_string()];
                    if let Some(rest) = resolve_args(gate, args, variables, qvm, functions) {
                        new_args.extend(rest);
                        apply_controlled_gate(qvm, gate, &new_args);
                    }
                } else {
                    eprintln!("Warning: only quantum gate calls are allowed inside qif");
                }
            }
            if let Some(else_cmds) = else_branch {
                qvm.apply_gate(&PauliX::new(), control);
                for cmd in else_cmds {
                    if let QLangCommand::ApplyGate(gate, args) = cmd {
                        let mut new_args = vec![control.to_string()];
                        if let Some(rest) = resolve_args(gate, args, variables, qvm, functions) {
                            new_args.extend(rest);
                            apply_controlled_gate(qvm, gate, &new_args);
                        }
                    }
                }
                qvm.apply_gate(&PauliX::new(), control);
            }
        }

        QLangCommand::ListFunctions => {
            let mut names: Vec<_> = functions.keys().collect();
            names.sort();
            println!("Available functions:");
            for name in names {
                if let Some(QLangCommand::FunctionDef { params, .. }) = functions.get(name) {
                    println!("  fn {}({})", name, params.join(", "));
                }
            }
        }

        QLangCommand::While { condition, body } => loop {
            if !evaluate_expression(condition, variables, qvm, functions).is_truthy() { break; }
            match run_ast(qvm, body, variables, functions) {
                ControlFlow::Break              => break,
                ControlFlow::Continue           => continue,
                ControlFlow::Return(v)          => return ControlFlow::Return(v),
                ControlFlow::None               => {}
            }
        },

        QLangCommand::For { var, start, end, body } => {
            let s = evaluate_expression(start, variables, qvm, functions).as_number() as i64;
            let e = evaluate_expression(end,   variables, qvm, functions).as_number() as i64;
            'outer: for i in s..e {
                variables.insert(var.clone(), Value::Number(i as f64));
                match run_ast(qvm, body, variables, functions) {
                    ControlFlow::Break    => break 'outer,
                    ControlFlow::Continue => continue 'outer,
                    ControlFlow::Return(v)=> return ControlFlow::Return(v),
                    ControlFlow::None     => {}
                }
            }
        }

        QLangCommand::Return(expr) => {
            let val = evaluate_expression(expr, variables, qvm, functions);
            return ControlFlow::Return(val);
        }

        QLangCommand::Break    => return ControlFlow::Break,
        QLangCommand::Continue => return ControlFlow::Continue,
    }
    ControlFlow::None
}

fn evaluate_expression(
    expr: &Expression,
    variables: &HashMap<String, Value>,
    qvm: &mut QVM,
    functions: &mut HashMap<String, QLangCommand>,
) -> Value {
    match expr {
        Expression::Number(n) => Value::Number(*n),
        Expression::Variable(name) => {
            variables.get(name).cloned().unwrap_or_else(|| {
                eprintln!("Warning: unknown variable '{}'", name);
                Value::Number(0.0)
            })
        }
        Expression::Measure(idx_expr) => {
            let q = evaluate_expression(idx_expr, variables, qvm, functions).as_qubit_index();
            Value::Bit(qvm.measure(q) as u8)
        }
        Expression::Array(exprs) => {
            Value::Array(exprs.iter().map(|e| evaluate_expression(e, variables, qvm, functions)).collect())
        }
        Expression::Index(arr_expr, idx_expr) => {
            let idx = evaluate_expression(idx_expr, variables, qvm, functions).as_qubit_index();
            match evaluate_expression(arr_expr, variables, qvm, functions) {
                Value::Array(arr) => {
                    if idx < arr.len() { arr[idx].clone() }
                    else {
                        eprintln!("Error: index {} out of bounds (length {})", idx, arr.len());
                        Value::Number(0.0)
                    }
                }
                _ => { eprintln!("Error: cannot index a non-array value"); Value::Number(0.0) }
            }
        }
        Expression::BinaryOp { left, op, right } => {
            let l = evaluate_expression(left,  variables, qvm, functions).as_number();
            let r = evaluate_expression(right, variables, qvm, functions).as_number();
            let v = match op {
                Operator::Add => l + r,
                Operator::Sub => l - r,
                Operator::Mul => l * r,
                Operator::Div => {
                    if r == 0.0 { eprintln!("Warning: division by zero"); f64::NAN } else { l / r }
                }
                Operator::Eq  => if (l - r).abs() < 1e-9 { 1.0 } else { 0.0 },
                Operator::Neq => if (l - r).abs() >= 1e-9 { 1.0 } else { 0.0 },
                Operator::Lt  => if l < r  { 1.0 } else { 0.0 },
                Operator::Gt  => if l > r  { 1.0 } else { 0.0 },
                Operator::Le  => if l <= r { 1.0 } else { 0.0 },
                Operator::Ge  => if l >= r { 1.0 } else { 0.0 },
                Operator::And => if l != 0.0 && r != 0.0 { 1.0 } else { 0.0 },
                Operator::Or  => if l != 0.0 || r != 0.0 { 1.0 } else { 0.0 },
            };
            Value::Number(v)
        }
        Expression::Call(name, args) => {
            if name == "alloc" {
                let idx = if let Some(arg) = args.first() {
                    evaluate_expression(arg, variables, qvm, functions).as_qubit_index()
                } else { 0 };
                return Value::Qubit(idx);
            }
            if let Some(QLangCommand::FunctionDef { params, body, .. }) = functions.get(name).cloned() {
                if args.len() != params.len() {
                    eprintln!("Error: '{}' expects {} arguments, got {}", name, params.len(), args.len());
                    return Value::Number(0.0);
                }
                let mut local_vars = HashMap::new();
                for (param, arg_expr) in params.iter().zip(args.iter()) {
                    let val = evaluate_expression(arg_expr, variables, qvm, functions);
                    local_vars.insert(param.clone(), val);
                }
                return match run_ast(qvm, &body, &mut local_vars, functions) {
                    ControlFlow::Return(v) => v,
                    ControlFlow::Break | ControlFlow::Continue => {
                        eprintln!("Warning: break/continue cannot escape a function body");
                        Value::Number(0.0)
                    }
                    ControlFlow::None => Value::Number(0.0),
                };
            }
            if let Some(resolved) = resolve_args(name, args, variables, qvm, functions) {
                apply_gate_dispatch(qvm, name, &resolved);
            }
            Value::Number(0.0)
        }
    }
}

fn resolve_args(
    gate_name: &str,
    args: &[Expression],
    variables: &HashMap<String, Value>,
    qvm: &mut QVM,
    functions: &mut HashMap<String, QLangCommand>,
) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for arg in args {
        match evaluate_expression(arg, variables, qvm, functions) {
            Value::Number(n) => out.push(n.to_string()),
            Value::Qubit(i)  => out.push(i.to_string()),
            Value::Bit(b)    => out.push(b.to_string()),
            Value::Array(_)  => {
                eprintln!("Error: gate '{}' does not accept array arguments", gate_name);
                return None;
            }
        }
    }
    Some(out)
}

fn apply_gate_dispatch(qvm: &mut QVM, name: &str, args: &[String]) {
    match name {
        "controlled_u" | "cu"  => apply_controlled_u(qvm, args),
        "hadamard" | "h"       => apply_one_q_gate(qvm, &Hadamard::new(), args),
        "identity"  | "id"     => apply_one_q_gate(qvm, &Identity::new(), args),
        "paulix" | "x"         => apply_one_q_gate(qvm, &PauliX::new(), args),
        "pauliy" | "y"         => apply_one_q_gate(qvm, &PauliY::new(), args),
        "pauliz" | "z"         => apply_one_q_gate(qvm, &PauliZ::new(), args),
        "s"                    => apply_one_q_gate(qvm, &S::new(), args),
        "sdagger" | "sdg"      => apply_one_q_gate(qvm, &SDagger::new(), args),
        "t"                    => apply_one_q_gate(qvm, &T::new(), args),
        "tdagger" | "tdg"      => apply_one_q_gate(qvm, &TDagger::new(), args),
        "phase"                => apply_one_q_with_1f64(qvm, &Phase::new, args),
        "rx"                   => apply_one_q_with_1f64(qvm, &RX::new, args),
        "ry"                   => apply_one_q_with_1f64(qvm, &RY::new, args),
        "rz"                   => apply_one_q_with_1f64(qvm, &RZ::new, args),
        "u1"                   => apply_one_q_with_1f64(qvm, &U1::new, args),
        "u2"                   => apply_one_q_with_2f64(qvm, &U2::new, args),
        "u3"                   => apply_one_q_with_3f64(qvm, &U3::new, args),
        "cnot" | "cx"          => apply_two_q_gate(qvm, &CNOT::new(), args),
        "iswap"                => apply_two_q_gate(qvm, &ISwap::new(), args),
        "swap"                 => apply_two_q_gate(qvm, &Swap::new(), args),
        "cy"                   => apply_two_q_gate(qvm, &ControlledY::new(), args),
        "cz"                   => apply_two_q_gate(qvm, &ControlledZ::new(), args),
        "toffoli"              => apply_three_q_gate(qvm, &Toffoli::new(), args),
        "fredkin"              => apply_three_q_gate(qvm, &Fredkin::new(), args),
        "alloc"                => {}
        _                      => eprintln!("Unknown gate: '{}'", name),
    }
}