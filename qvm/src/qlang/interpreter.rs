use std::collections::HashMap;
use std::fmt;

use crate::gates::{
    one_q::{
        hadamard::*, identity::*, pauli_x::*, pauli_y::*, pauli_z::*, s::*, s_dagger::*, t::*,
        t_dagger::*,
    },
    rotation_q::{phase::*, rx::*, ry::*, rz::*, u1::*, u2::*, u3::*},
    three_q::{fredkin::*, toffoli::*},
    two_q::{cnot::*, cy::*, cz::*, iswap::*, swap::*},
};
use crate::qlang::{apply::*, ast::{Expression, Operator, QLangCommand}};
use crate::qvm::QVM;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Array(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
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
            _ => panic!("Expected number, found array"),
        }
    }
    
    pub fn as_usize(&self) -> usize {
        self.as_number() as usize
    }
}

/// Executes a sequence of QLang commands (AST) on a given quantum virtual machine.
pub fn run_ast(
    qvm: &mut QVM,
    ast: &[QLangCommand],
    variables: &mut HashMap<String, Value>,
    functions: &mut HashMap<String, QLangCommand>,
) {
    for cmd in ast {
        execute_command(qvm, cmd, variables, functions);
    }
}

fn execute_command(
    qvm: &mut QVM,
    cmd: &QLangCommand,
    variables: &mut HashMap<String, Value>,
    functions: &mut HashMap<String, QLangCommand>,
) {
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
            // Check if it's a user-defined function
            if let Some(QLangCommand::FunctionDef { params, body, .. }) = functions.get(name).cloned() {
                if args.len() != params.len() {
                    println!("Error: Function {} expects {} arguments, got {}", name, params.len(), args.len());
                    return;
                }

                // Create local scope initialized with argument values
                let mut local_vars = HashMap::new();
                for (param, arg_expr) in params.iter().zip(args.iter()) {
                    // Evaluate argument in the CALLER's scope
                    let val = evaluate_expression(arg_expr, variables, qvm);
                    local_vars.insert(param.clone(), val);
                }

                run_ast(qvm, &body, &mut local_vars, functions);
            } else {
                // Standard Gate
                // Resolve arguments (expressions to strings/values)
                // Native gates expect strings (variable names or numbers).
                // But now we have evaluated values.
                // We need to convert values to strings?
                // apply_gate_dispatch expects Vec<String>.
                // If value is Number, convert to string.
                // If value is Array, we can't pass it to standard gate (unless gate supports it, e.g. measure_many).
                // But standard gates like h(q) expect q as index.
                
                let mut resolved_args = Vec::new();
                for arg_expr in args {
                    let val = evaluate_expression(arg_expr, variables, qvm);
                    match val {
                        Value::Number(n) => resolved_args.push(n.to_string()),
                        Value::Array(_) => {
                             println!("Error: Standard gate {} does not support array arguments directly", name);
                             return;
                        }
                    }
                }
                apply_gate_dispatch(qvm, name, &resolved_args);
            }
        }
        QLangCommand::Display => qvm.display(),
        QLangCommand::MeasureAll => {
            qvm.measure_all();
        }
        QLangCommand::Measure(q) => {
            qvm.measure(*q);
        }
        QLangCommand::MeasureMany(qs) => {
            qvm.measure_many(qs);
        }
        QLangCommand::Let { name, value } | QLangCommand::Assign { name, value } => {
            let val = evaluate_expression(value, variables, qvm);
            variables.insert(name.clone(), val);
        }
        QLangCommand::If { condition, then_branch, else_branch } => {
            let cond_val = evaluate_expression(condition, variables, qvm).as_number();
            if cond_val != 0.0 {
                run_ast(qvm, then_branch, variables, functions);
            } else if let Some(else_cmds) = else_branch {
                run_ast(qvm, else_cmds, variables, functions);
            }
        }
        QLangCommand::ListFunctions => {
            println!("Available Functions:");
            let mut names: Vec<_> = functions.keys().collect();
            names.sort();
            for name in names {
                if let Some(QLangCommand::FunctionDef { params, .. }) = functions.get(name) {
                    println!("  fn {}({})", name, params.join(", "));
                }
            }
        }
        QLangCommand::While { condition, body } => {
            while evaluate_expression(condition, variables, qvm).as_number() != 0.0 {
                run_ast(qvm, body, variables, functions);
            }
        }
        QLangCommand::For { var, start, end, body } => {
            let start_val = evaluate_expression(start, variables, qvm).as_number() as i64;
            let end_val = evaluate_expression(end, variables, qvm).as_number() as i64;
            
            for i in start_val..end_val {
                variables.insert(var.clone(), Value::Number(i as f64));
                run_ast(qvm, body, variables, functions);
            }
        }
    }
}

fn evaluate_expression(expr: &Expression, variables: &HashMap<String, Value>, qvm: &mut QVM) -> Value {
    match expr {
        Expression::Number(n) => Value::Number(*n),
        Expression::Variable(name) => variables.get(name).cloned().unwrap_or(Value::Number(0.0)),
        Expression::Measure(idx_expr) => {
            let idx = evaluate_expression(idx_expr, variables, qvm).as_usize();
            Value::Number(qvm.measure(idx) as f64)
        }
        Expression::Array(exprs) => {
            let vals = exprs.iter().map(|e| evaluate_expression(e, variables, qvm)).collect();
            Value::Array(vals)
        }
        Expression::Index(arr_expr, idx_expr) => {
            let arr_val = evaluate_expression(arr_expr, variables, qvm);
            let idx_val = evaluate_expression(idx_expr, variables, qvm).as_usize();
            
            if let Value::Array(arr) = arr_val {
                if idx_val < arr.len() {
                    arr[idx_val].clone()
                } else {
                    panic!("Index out of bounds: {} >= {}", idx_val, arr.len());
                }
            } else {
                panic!("Cannot index non-array value");
            }
        }
        Expression::BinaryOp { left, op, right } => {
            let l = evaluate_expression(left, variables, qvm).as_number();
            let r = evaluate_expression(right, variables, qvm).as_number();
            let res = match op {
                Operator::Add => l + r,
                Operator::Sub => l - r,
                Operator::Mul => l * r,
                Operator::Div => l / r,
                Operator::Eq => if (l - r).abs() < 1e-9 { 1.0 } else { 0.0 },
                Operator::Neq => if (l - r).abs() >= 1e-9 { 1.0 } else { 0.0 },
                Operator::Lt => if l < r { 1.0 } else { 0.0 },
                Operator::Gt => if l > r { 1.0 } else { 0.0 },
                Operator::Le => if l <= r { 1.0 } else { 0.0 },
                Operator::Ge => if l >= r { 1.0 } else { 0.0 },
                Operator::And => if l != 0.0 && r != 0.0 { 1.0 } else { 0.0 },
                Operator::Or => if l != 0.0 || r != 0.0 { 1.0 } else { 0.0 },
            };
            Value::Number(res)
        }
    }
}

/// Dispatches a gate application by matching the gate name to its implementation.
fn apply_gate_dispatch(qvm: &mut QVM, name: &str, args: &[String]) {
    match name {
        "controlled_u" | "cu" => apply_controlled_u(qvm, args),
        "hadamard" | "h" => apply_one_q_gate(qvm, &Hadamard::new(), args),
        "identity" | "id" => apply_one_q_gate(qvm, &Identity::new(), args),
        "paulix" | "x" => apply_one_q_gate(qvm, &PauliX::new(), args),
        "pauliy" | "y" => apply_one_q_gate(qvm, &PauliY::new(), args),
        "pauliz" | "z" => apply_one_q_gate(qvm, &PauliZ::new(), args),
        "s" => apply_one_q_gate(qvm, &S::new(), args),
        "sdagger" | "sdg" => apply_one_q_gate(qvm, &SDagger::new(), args),
        "t" => apply_one_q_gate(qvm, &T::new(), args),
        "tdagger" | "tdg" => apply_one_q_gate(qvm, &TDagger::new(), args),
        "phase" => apply_one_q_with_1f64(qvm, &Phase::new, args),
        "rx" => apply_one_q_with_1f64(qvm, &RX::new, args),
        "ry" => apply_one_q_with_1f64(qvm, &RY::new, args),
        "rz" => apply_one_q_with_1f64(qvm, &RZ::new, args),
        "u1" => apply_one_q_with_1f64(qvm, &U1::new, args),
        "u2" => apply_one_q_with_2f64(qvm, &U2::new, args),
        "u3" => apply_one_q_with_3f64(qvm, &U3::new, args),
        "cnot" | "cx" => apply_two_q_gate(qvm, &CNOT::new(), args),
        "iswap" => apply_two_q_gate(qvm, &ISwap::new(), args),
        "swap" => apply_two_q_gate(qvm, &Swap::new(), args),
        "cy" => apply_two_q_gate(qvm, &ControlledY::new(), args),
        "cz" => apply_two_q_gate(qvm, &ControlledZ::new(), args),
        "toffoli" => apply_three_q_gate(qvm, &Toffoli::new(), args),
        "fredkin" => apply_three_q_gate(qvm, &Fredkin::new(), args),
        _ => println!("Gate desconhecido: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qvm::QVM;

    #[test]
    fn test_hadamard_application() {
        let mut qvm = QVM::new(1);
        let mut vars = HashMap::new();
        let mut funcs = HashMap::new();
        let ast = vec![
            QLangCommand::Create(1),
            QLangCommand::ApplyGate("hadamard".into(), vec!["0".into()]),
        ];
        run_ast(&mut qvm, &ast, &mut vars, &mut funcs);

        let state = qvm.state_vector();
        let norm = (1.0 / 2.0f64).sqrt();
        assert!((state[0].norm_sqr() - norm.powi(2)).abs() < 1e-6);
        assert!((state[1].norm_sqr() - norm.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_variable_assignment() {
        let mut qvm = QVM::new(1);
        let mut vars = HashMap::new();
        let mut funcs = HashMap::new();
        let ast = vec![
            QLangCommand::Let { name: "a".into(), value: Expression::Number(10.0) },
        ];
        run_ast(&mut qvm, &ast, &mut vars, &mut funcs);
        assert_eq!(*vars.get("a").unwrap(), 10.0);
    }

    #[test]
    fn test_if_condition() {
        let mut qvm = QVM::new(1);
        let mut vars = HashMap::new();
        let mut funcs = HashMap::new();
        // let a = 1; if (a == 1) { x(0) }
        let ast = vec![
            QLangCommand::Let { name: "a".into(), value: Expression::Number(1.0) },
            QLangCommand::If {
                condition: Expression::BinaryOp {
                    left: Box::new(Expression::Variable("a".into())),
                    op: Operator::Eq,
                    right: Box::new(Expression::Number(1.0))
                },
                then_branch: vec![QLangCommand::ApplyGate("x".into(), vec!["0".into()])],
                else_branch: None
            }
        ];
        run_ast(&mut qvm, &ast, &mut vars, &mut funcs);
        
        // Check if X gate was applied (state |1>)
        let state = qvm.state_vector();
        assert!((state[1].norm_sqr() - 1.0).abs() < 1e-6);
    }
}
