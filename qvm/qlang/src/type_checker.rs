use std::collections::{HashMap, HashSet};

use crate::ast::{Expression, Operator, QLangCommand, TypeAnnotation};
use crate::errors::QLangError;

#[derive(Clone, Debug, PartialEq)]
pub enum QLangType {
    Qubit,
    Bit,
    Int,
    Float,
    Bool,
    Array(Box<QLangType>),
    Void,
    Unknown,
}

impl QLangType {
    pub fn from_annotation(ann: &TypeAnnotation) -> Self {
        match ann {
            TypeAnnotation::Qubit       => QLangType::Qubit,
            TypeAnnotation::Bit         => QLangType::Bit,
            TypeAnnotation::Int         => QLangType::Int,
            TypeAnnotation::Float       => QLangType::Float,
            TypeAnnotation::Bool        => QLangType::Bool,
            TypeAnnotation::Void        => QLangType::Void,
            TypeAnnotation::Array(inner)=>
                QLangType::Array(Box::new(QLangType::from_annotation(inner))),
        }
    }
}

impl std::fmt::Display for QLangType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QLangType::Qubit    => write!(f, "qubit"),
            QLangType::Bit      => write!(f, "bit"),
            QLangType::Int      => write!(f, "int"),
            QLangType::Float    => write!(f, "float"),
            QLangType::Bool     => write!(f, "bool"),
            QLangType::Array(t) => write!(f, "[{}]", t),
            QLangType::Void     => write!(f, "void"),
            QLangType::Unknown  => write!(f, "unknown"),
        }
    }
}

pub struct TypeChecker {
    env: HashMap<String, QLangType>,
    consumed_slots: HashSet<usize>,
    qubit_vars: HashMap<String, usize>,
    moved_vars: HashSet<String>,
    functions: HashMap<String, (usize, Vec<QLangType>, QLangType)>,
    num_qubits: usize,
    in_qif: bool,
    pub errors: Vec<QLangError>,
}

impl TypeChecker {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            env: HashMap::new(),
            consumed_slots: HashSet::new(),
            qubit_vars: HashMap::new(),
            moved_vars: HashSet::new(),
            functions: HashMap::new(),
            num_qubits,
            in_qif: false,
            errors: Vec::new(),
        }
    }

    pub fn check(&mut self, ast: &[QLangCommand]) -> Vec<QLangError> {
        self.errors.clear();
        for cmd in ast { self.check_command(cmd); }
        self.errors.clone()
    }

    fn check_command(&mut self, cmd: &QLangCommand) {
        match cmd {
            QLangCommand::Create(n) => {
                self.num_qubits = *n;
                self.consumed_slots.clear();
                self.qubit_vars.clear();
                self.moved_vars.clear();
                self.env.clear();
            }
            QLangCommand::Let { name, type_ann, value } => {
                let inferred = self.infer_type(value);
                let declared = type_ann.as_ref().map(QLangType::from_annotation);
                if let (Some(decl), ref inf) = (&declared, &inferred) {
                    if !types_compatible(decl, inf) {
                        self.errors.push(QLangError::TypeError(format!(
                            "variable '{}' declared as {} but assigned {}", name, decl, inf
                        )));
                    }
                }
                let final_type = declared.unwrap_or(inferred.clone());
                if let QLangType::Qubit = &final_type {
                    if let Some(slot) = extract_qubit_slot(value) {
                        self.qubit_vars.insert(name.clone(), slot);
                    }
                }
                if let Expression::Variable(src) = value {
                    if self.env.get(src) == Some(&QLangType::Qubit) {
                        self.moved_vars.insert(src.clone());
                    }
                }
                self.env.insert(name.clone(), final_type);
            }
            QLangCommand::Assign { name, value } => {
                let inferred = self.infer_type(value);
                if let Expression::Variable(src) = value {
                    if self.env.get(src) == Some(&QLangType::Qubit) {
                        self.moved_vars.insert(src.clone());
                    }
                }
                self.env.insert(name.clone(), inferred);
            }
            QLangCommand::FunctionDef { name, params, param_types, return_type, body } => {
                let p_types: Vec<QLangType> = param_types.iter()
                    .map(|t| t.as_ref().map(QLangType::from_annotation).unwrap_or(QLangType::Unknown))
                    .collect();
                let r_type = return_type.as_ref().map(QLangType::from_annotation).unwrap_or(QLangType::Void);
                self.functions.insert(name.clone(), (params.len(), p_types.clone(), r_type));
                let mut inner = TypeChecker {
                    env: params.iter().zip(p_types.iter()).map(|(p, t)| (p.clone(), t.clone())).collect(),
                    consumed_slots: self.consumed_slots.clone(),
                    qubit_vars: HashMap::new(),
                    moved_vars: HashSet::new(),
                    functions: self.functions.clone(),
                    num_qubits: self.num_qubits,
                    in_qif: false,
                    errors: Vec::new(),
                };
                inner.check(body);
                self.errors.extend(inner.errors);
            }
            QLangCommand::Measure(q) => self.check_qubit_slot(*q, true),
            QLangCommand::MeasureMany(qs) => { for &q in qs { self.check_qubit_slot(q, true); } }
            QLangCommand::MeasureAll => { for i in 0..self.num_qubits { self.consumed_slots.insert(i); } }
            QLangCommand::ApplyGate(name, args) => {
                for arg in args { self.check_expression_qubit_use(arg); }
                if let Some((expected, _, _)) = self.functions.get(name) {
                    if args.len() != *expected {
                        self.errors.push(QLangError::ArityMismatch { name: name.clone(), expected: *expected, got: args.len() });
                    }
                }
            }
            QLangCommand::QIf { condition, then_branch, else_branch } => {
                self.check_expression_qubit_use(condition);
                let prev = self.in_qif;
                self.in_qif = true;
                for cmd in then_branch { self.check_qif_cmd(cmd); }
                if let Some(else_cmds) = else_branch { for cmd in else_cmds { self.check_qif_cmd(cmd); } }
                self.in_qif = prev;
            }
            QLangCommand::If { then_branch, else_branch, .. } => {
                for cmd in then_branch { self.check_command(cmd); }
                if let Some(cmds) = else_branch { for cmd in cmds { self.check_command(cmd); } }
            }
            QLangCommand::While { body, .. } => { for cmd in body { self.check_command(cmd); } }
            QLangCommand::For { var, body, .. } => {
                self.env.insert(var.clone(), QLangType::Int);
                for cmd in body { self.check_command(cmd); }
            }
            QLangCommand::Return(expr) => { self.infer_type(expr); }
            QLangCommand::Display | QLangCommand::ListFunctions | QLangCommand::Import { .. }
            | QLangCommand::Break | QLangCommand::Continue => {}
        }
    }

    fn check_qif_cmd(&mut self, cmd: &QLangCommand) {
        match cmd {
            QLangCommand::Measure(_) | QLangCommand::MeasureMany(_) | QLangCommand::MeasureAll => {
                self.errors.push(QLangError::NonGateInQif);
            }
            QLangCommand::ApplyGate(_, args) => { for arg in args { self.check_expression_qubit_use(arg); } }
            other => self.check_command(other),
        }
    }

    fn check_expression_qubit_use(&mut self, expr: &Expression) {
        match expr {
            Expression::Number(n) => {
                let slot = *n as usize;
                if self.consumed_slots.contains(&slot) { self.errors.push(QLangError::QubitAlreadyMeasured(slot)); }
            }
            Expression::Variable(v) => {
                if self.moved_vars.contains(v) {
                    self.errors.push(QLangError::TypeError(format!("qubit variable '{}' has been moved and cannot be used again", v)));
                }
                if let Some(&slot) = self.qubit_vars.get(v) {
                    if self.consumed_slots.contains(&slot) { self.errors.push(QLangError::QubitAlreadyMeasured(slot)); }
                }
            }
            Expression::Call(_, args) | Expression::Array(args) => { for a in args { self.check_expression_qubit_use(a); } }
            Expression::Index(arr, idx) => { self.check_expression_qubit_use(arr); self.check_expression_qubit_use(idx); }
            Expression::Measure(inner) => {
                if let Expression::Number(n) = inner.as_ref() { self.check_qubit_slot(*n as usize, true); }
                self.check_expression_qubit_use(inner);
            }
            Expression::BinaryOp { left, right, .. } => { self.check_expression_qubit_use(left); self.check_expression_qubit_use(right); }
        }
    }

    fn check_qubit_slot(&mut self, slot: usize, consuming: bool) {
        if slot >= self.num_qubits { self.errors.push(QLangError::QubitOutOfRange(slot)); return; }
        if self.consumed_slots.contains(&slot) { self.errors.push(QLangError::QubitAlreadyMeasured(slot)); }
        else if consuming { self.consumed_slots.insert(slot); }
    }

    pub fn infer_type(&self, expr: &Expression) -> QLangType {
        match expr {
            Expression::Number(n) => { if *n == n.floor() { QLangType::Int } else { QLangType::Float } }
            Expression::Variable(name) => { self.env.get(name).cloned().unwrap_or(QLangType::Unknown) }
            Expression::Measure(_) => QLangType::Bit,
            Expression::Array(es) => {
                let elem = if es.is_empty() { QLangType::Unknown } else { self.infer_type(&es[0]) };
                QLangType::Array(Box::new(elem))
            }
            Expression::Index(arr, _) => {
                if let QLangType::Array(elem) = self.infer_type(arr) { *elem } else { QLangType::Unknown }
            }
            Expression::BinaryOp { left, op, right } => match op {
                Operator::Eq | Operator::Neq | Operator::Lt | Operator::Gt
                | Operator::Le | Operator::Ge | Operator::And | Operator::Or => QLangType::Bool,
                _ => match (self.infer_type(left), self.infer_type(right)) {
                    (QLangType::Float, _) | (_, QLangType::Float) => QLangType::Float,
                    (QLangType::Int, QLangType::Int) => QLangType::Int,
                    _ => QLangType::Unknown,
                },
            },
            Expression::Call(name, _) => {
                if name == "alloc" { return QLangType::Qubit; }
                self.functions.get(name).map(|(_, _, ret)| ret.clone()).unwrap_or(QLangType::Unknown)
            }
        }
    }
}

fn types_compatible(a: &QLangType, b: &QLangType) -> bool {
    a == b || matches!(a, QLangType::Unknown) || matches!(b, QLangType::Unknown)
}

fn extract_qubit_slot(expr: &Expression) -> Option<usize> {
    match expr {
        Expression::Number(n) => Some(*n as usize),
        Expression::Call(name, args) if name == "alloc" => {
            args.first().and_then(|a| { if let Expression::Number(n) = a { Some(*n as usize) } else { None } })
        }
        _ => None,
    }
}