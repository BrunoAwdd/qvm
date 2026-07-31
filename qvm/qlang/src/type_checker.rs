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
    pub fn from_annotation(annotation: &TypeAnnotation) -> Self {
        match annotation {
            TypeAnnotation::Qubit => Self::Qubit,
            TypeAnnotation::Bit => Self::Bit,
            TypeAnnotation::Int => Self::Int,
            TypeAnnotation::Float => Self::Float,
            TypeAnnotation::Bool => Self::Bool,
            TypeAnnotation::Void => Self::Void,
            TypeAnnotation::Array(inner) => Self::Array(Box::new(Self::from_annotation(inner))),
        }
    }
}

impl std::fmt::Display for QLangType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qubit => write!(f, "qubit"),
            Self::Bit => write!(f, "bit"),
            Self::Int => write!(f, "int"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::Array(inner) => write!(f, "[{}]", inner),
            Self::Void => write!(f, "void"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Clone)]
struct FunctionSignature {
    params: Vec<QLangType>,
    return_type: QLangType,
}

#[derive(Clone)]
pub struct TypeChecker {
    env: HashMap<String, QLangType>,
    consumed_slots: HashSet<usize>,
    qubit_vars: HashMap<String, usize>,
    slot_owners: HashMap<usize, String>,
    moved_vars: HashSet<String>,
    consumed_vars: HashSet<String>,
    functions: HashMap<String, FunctionSignature>,
    num_qubits: usize,
    expected_return: Option<QLangType>,
    loop_depth: usize,
    in_qif: bool,
    qif_control: Option<Expression>,
    pub errors: Vec<QLangError>,
}

impl TypeChecker {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            env: HashMap::new(),
            consumed_slots: HashSet::new(),
            qubit_vars: HashMap::new(),
            slot_owners: HashMap::new(),
            moved_vars: HashSet::new(),
            consumed_vars: HashSet::new(),
            functions: HashMap::new(),
            num_qubits,
            expected_return: None,
            loop_depth: 0,
            in_qif: false,
            qif_control: None,
            errors: Vec::new(),
        }
    }

    pub fn check(&mut self, ast: &[QLangCommand]) -> Vec<QLangError> {
        self.errors.clear();
        self.env.clear();
        self.consumed_slots.clear();
        self.qubit_vars.clear();
        self.slot_owners.clear();
        self.moved_vars.clear();
        self.consumed_vars.clear();
        self.functions.clear();
        self.predeclare_functions(ast);
        for command in ast {
            self.check_command(command);
        }
        self.errors.clone()
    }

    fn predeclare_functions(&mut self, ast: &[QLangCommand]) {
        for command in ast {
            if let QLangCommand::FunctionDef {
                name,
                params,
                param_types,
                return_type,
                ..
            } = command
            {
                let types = params
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        param_types
                            .get(index)
                            .and_then(Option::as_ref)
                            .map(QLangType::from_annotation)
                            .unwrap_or(QLangType::Unknown)
                    })
                    .collect();
                let return_type = return_type
                    .as_ref()
                    .map(QLangType::from_annotation)
                    .unwrap_or(QLangType::Void);
                self.functions.insert(
                    name.clone(),
                    FunctionSignature {
                        params: types,
                        return_type,
                    },
                );
            }
        }
    }

    fn check_command(&mut self, command: &QLangCommand) {
        if self.in_qif && !matches!(command, QLangCommand::ApplyGate(name, _) if is_qif_gate(name))
        {
            self.errors.push(QLangError::NonGateInQif);
            return;
        }
        match command {
            QLangCommand::Create(size) => self.reset_register(*size),
            QLangCommand::Let {
                name,
                type_ann,
                value,
            } => self.check_let(name, type_ann.as_ref(), value),
            QLangCommand::Assign { name, value } => self.check_assignment(name, value),
            QLangCommand::FunctionDef {
                name, params, body, ..
            } => self.check_function(name, params, body),
            QLangCommand::ApplyGate(name, args) => {
                self.check_call(name, args, true);
            }
            QLangCommand::Measure(slot) => self.check_qubit_slot(*slot, true),
            QLangCommand::MeasureMany(slots) => {
                for slot in slots {
                    self.check_qubit_slot(*slot, true);
                }
            }
            QLangCommand::MeasureAll => {
                self.consumed_slots.extend(0..self.num_qubits);
            }
            QLangCommand::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_condition(condition, "if");
                self.check_branches(then_branch, else_branch.as_deref());
            }
            QLangCommand::QIf {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_type = self.check_expression(condition);
                if !matches!(
                    condition_type,
                    QLangType::Qubit | QLangType::Int | QLangType::Unknown
                ) {
                    self.type_error(format!(
                        "qif condition must be a qubit, found {condition_type}"
                    ));
                }
                let previous = self.in_qif;
                let previous_control = self.qif_control.replace(condition.clone());
                self.in_qif = true;
                self.check_branches(then_branch, else_branch.as_deref());
                self.in_qif = previous;
                self.qif_control = previous_control;
            }
            QLangCommand::While { condition, body } => {
                self.check_condition(condition, "while");
                self.check_loop(body);
            }
            QLangCommand::For {
                var,
                start,
                end,
                body,
            } => {
                self.expect_type(start, &QLangType::Int, "for range start");
                self.expect_type(end, &QLangType::Int, "for range end");
                let previous = self.env.insert(var.clone(), QLangType::Int);
                self.check_loop(body);
                if let Some(previous) = previous {
                    self.env.insert(var.clone(), previous);
                } else {
                    self.env.remove(var);
                }
            }
            QLangCommand::Return(expression) => self.check_return(expression),
            QLangCommand::Break | QLangCommand::Continue => {
                if self.loop_depth == 0 {
                    self.type_error("break/continue may only be used inside a loop".into());
                }
            }
            QLangCommand::Display | QLangCommand::ListFunctions | QLangCommand::Import { .. } => {}
        }
    }

    fn reset_register(&mut self, size: usize) {
        self.num_qubits = size;
        self.env.clear();
        self.consumed_slots.clear();
        self.qubit_vars.clear();
        self.slot_owners.clear();
        self.moved_vars.clear();
        self.consumed_vars.clear();
    }

    fn check_let(&mut self, name: &str, annotation: Option<&TypeAnnotation>, value: &Expression) {
        let inferred = self.check_expression(value);
        let declared = annotation.map(QLangType::from_annotation);
        if let Some(declared) = &declared {
            if !types_compatible(declared, &inferred) {
                self.type_error(format!(
                    "variable '{name}' declared as {declared} but assigned {inferred}"
                ));
            }
        }
        let final_type = declared.unwrap_or(inferred);
        self.bind_value(name, value, &final_type);
        self.env.insert(name.to_string(), final_type);
    }

    fn check_assignment(&mut self, name: &str, value: &Expression) {
        let Some(existing) = self.env.get(name).cloned() else {
            self.errors
                .push(QLangError::UnknownVariable(name.to_string()));
            self.check_expression(value);
            return;
        };
        let inferred = self.check_expression(value);
        if !types_compatible(&existing, &inferred) {
            self.type_error(format!(
                "cannot assign {inferred} to variable '{name}' of type {existing}"
            ));
            return;
        }
        self.release_qubit_binding(name);
        self.bind_value(name, value, &existing);
    }

    fn bind_value(&mut self, name: &str, value: &Expression, value_type: &QLangType) {
        if value_type != &QLangType::Qubit {
            return;
        }
        let slot = match value {
            Expression::Variable(source) => {
                let slot = self.qubit_vars.get(source).copied();
                if slot.is_some() {
                    self.moved_vars.insert(source.clone());
                    self.qubit_vars.remove(source);
                }
                if self.consumed_vars.contains(source) {
                    self.consumed_vars.insert(name.to_string());
                }
                slot
            }
            _ => extract_qubit_slot(value),
        };
        let Some(slot) = slot else { return };
        if slot >= self.num_qubits {
            self.errors.push(QLangError::QubitOutOfRange(slot));
            return;
        }
        if let Some(owner) = self.slot_owners.get(&slot) {
            if owner != name && !self.moved_vars.contains(owner) {
                self.type_error(format!(
                    "qubit {slot} is already owned by variable '{owner}'"
                ));
                return;
            }
        }
        self.slot_owners.insert(slot, name.to_string());
        self.qubit_vars.insert(name.to_string(), slot);
        self.moved_vars.remove(name);
    }

    fn release_qubit_binding(&mut self, name: &str) {
        if let Some(slot) = self.qubit_vars.remove(name) {
            if self
                .slot_owners
                .get(&slot)
                .is_some_and(|owner| owner == name)
            {
                self.slot_owners.remove(&slot);
            }
        }
    }

    fn check_function(&mut self, name: &str, params: &[String], body: &[QLangCommand]) {
        let Some(signature) = self.functions.get(name).cloned() else {
            return;
        };
        let mut inner = Self::new(self.num_qubits);
        inner.functions = self.functions.clone();
        inner.expected_return = Some(signature.return_type.clone());
        for (param, param_type) in params.iter().zip(signature.params.iter()) {
            inner.env.insert(param.clone(), param_type.clone());
        }
        for command in body {
            inner.check_command(command);
        }
        if signature.return_type != QLangType::Void && !guarantees_return(body) {
            inner.type_error(format!(
                "function '{name}' declares return type {} but does not return on every path",
                signature.return_type
            ));
        }
        self.errors.extend(inner.errors);
    }

    fn check_call(&mut self, name: &str, args: &[Expression], statement: bool) -> QLangType {
        if self.in_qif && !is_qif_gate(name) {
            self.errors.push(QLangError::NonGateInQif);
        }
        if let Some(signature) = self.functions.get(name).cloned() {
            if args.len() != signature.params.len() {
                self.errors.push(QLangError::ArityMismatch {
                    name: name.to_string(),
                    expected: signature.params.len(),
                    got: args.len(),
                });
            }
            for (index, argument) in args.iter().enumerate() {
                let actual = self.check_expression(argument);
                if let Some(expected) = signature.params.get(index) {
                    if !gate_argument_compatible(expected, &actual) {
                        self.type_error(format!(
                            "argument {} of '{name}' expects {expected}, found {actual}",
                            index + 1
                        ));
                    }
                }
            }
            return signature.return_type;
        }
        if name == "alloc" {
            if args.len() != 1 {
                self.errors.push(QLangError::ArityMismatch {
                    name: name.into(),
                    expected: 1,
                    got: args.len(),
                });
            }
            for argument in args {
                self.expect_type(argument, &QLangType::Int, "alloc argument");
            }
            return QLangType::Qubit;
        }
        if let Some(parameter_types) = native_gate_signature(name) {
            if args.len() != parameter_types.len() {
                self.errors.push(QLangError::ArityMismatch {
                    name: name.into(),
                    expected: parameter_types.len(),
                    got: args.len(),
                });
            }
            for (index, argument) in args.iter().enumerate() {
                let actual = self.check_expression(argument);
                if let Some(expected) = parameter_types.get(index) {
                    if expected == &QLangType::Qubit {
                        self.check_qubit_argument(argument);
                    }
                    if !gate_argument_compatible(expected, &actual) {
                        self.type_error(format!(
                            "argument {} of gate '{name}' expects {expected}, found {actual}",
                            index + 1
                        ));
                    }
                }
            }
            self.check_distinct_gate_qubits(name, args, &parameter_types);
            return QLangType::Void;
        }
        for argument in args {
            self.check_expression(argument);
        }
        self.errors.push(QLangError::UnknownGate(name.to_string()));
        if statement {
            QLangType::Void
        } else {
            QLangType::Unknown
        }
    }

    fn check_expression(&mut self, expression: &Expression) -> QLangType {
        match expression {
            Expression::Number(number) => {
                if number.is_finite() && *number == number.floor() {
                    QLangType::Int
                } else {
                    QLangType::Float
                }
            }
            Expression::Variable(name) => {
                if self.moved_vars.contains(name) {
                    self.type_error(format!(
                        "qubit variable '{name}' has been moved and cannot be used again"
                    ));
                }
                if self.consumed_vars.contains(name) {
                    self.type_error(format!("qubit variable '{name}' has already been measured"));
                }
                self.env.get(name).cloned().unwrap_or_else(|| {
                    self.errors.push(QLangError::UnknownVariable(name.clone()));
                    QLangType::Unknown
                })
            }
            Expression::Measure(inner) => {
                let inner_type = self.check_expression(inner);
                if !matches!(
                    inner_type,
                    QLangType::Qubit | QLangType::Int | QLangType::Unknown
                ) {
                    self.type_error(format!("measure expects a qubit, found {inner_type}"));
                }
                self.consume_qubit_argument(inner);
                QLangType::Bit
            }
            Expression::Array(elements) => self.check_array(elements),
            Expression::Index(array, index) => {
                self.expect_type(index, &QLangType::Int, "array index");
                match self.check_expression(array) {
                    QLangType::Array(element) => *element,
                    QLangType::Unknown => QLangType::Unknown,
                    other => {
                        self.type_error(format!("cannot index value of type {other}"));
                        QLangType::Unknown
                    }
                }
            }
            Expression::BinaryOp { left, op, right } => self.check_binary(left, op, right),
            Expression::Call(name, args) => self.check_call(name, args, false),
        }
    }

    fn check_array(&mut self, elements: &[Expression]) -> QLangType {
        let Some(first) = elements.first() else {
            return QLangType::Array(Box::new(QLangType::Unknown));
        };
        let element_type = self.check_expression(first);
        for element in &elements[1..] {
            let actual = self.check_expression(element);
            if !types_compatible(&element_type, &actual) {
                self.type_error(format!(
                    "array elements must have one type, found {element_type} and {actual}"
                ));
            }
        }
        QLangType::Array(Box::new(element_type))
    }

    fn check_binary(
        &mut self,
        left: &Expression,
        operator: &Operator,
        right: &Expression,
    ) -> QLangType {
        let left_type = self.check_expression(left);
        let right_type = self.check_expression(right);
        match operator {
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                if !is_numeric(&left_type) || !is_numeric(&right_type) {
                    self.type_error(format!(
                        "operator '{operator}' requires numbers, found {left_type} and {right_type}"
                    ));
                    QLangType::Unknown
                } else if left_type == QLangType::Float || right_type == QLangType::Float {
                    QLangType::Float
                } else {
                    QLangType::Int
                }
            }
            Operator::And | Operator::Or => {
                if !is_condition_type(&left_type) || !is_condition_type(&right_type) {
                    self.type_error(format!(
                        "operator '{operator}' requires boolean-compatible operands"
                    ));
                }
                QLangType::Bool
            }
            Operator::Eq | Operator::Neq => {
                if !types_compatible(&left_type, &right_type) {
                    self.type_error(format!("cannot compare {left_type} with {right_type}"));
                }
                QLangType::Bool
            }
            Operator::Lt | Operator::Gt | Operator::Le | Operator::Ge => {
                if !is_numeric(&left_type) || !is_numeric(&right_type) {
                    self.type_error(format!("operator '{operator}' requires numeric operands"));
                }
                QLangType::Bool
            }
        }
    }

    fn check_condition(&mut self, condition: &Expression, context: &str) {
        let condition_type = self.check_expression(condition);
        if !is_condition_type(&condition_type) {
            self.type_error(format!(
                "{context} condition must be boolean-compatible, found {condition_type}"
            ));
        }
    }

    fn check_branches(
        &mut self,
        then_branch: &[QLangCommand],
        else_branch: Option<&[QLangCommand]>,
    ) {
        let initial_errors = self.errors.len();
        let mut then_checker = self.clone();
        then_checker.errors.clear();
        for command in then_branch {
            then_checker.check_command(command);
        }
        let mut else_checker = self.clone();
        else_checker.errors.clear();
        if let Some(branch) = else_branch {
            for command in branch {
                else_checker.check_command(command);
            }
        }
        self.errors.truncate(initial_errors);
        self.errors.extend(then_checker.errors);
        self.errors.extend(else_checker.errors);
        self.consumed_slots.extend(then_checker.consumed_slots);
        self.consumed_slots.extend(else_checker.consumed_slots);
        self.moved_vars.extend(then_checker.moved_vars);
        self.moved_vars.extend(else_checker.moved_vars);
        self.consumed_vars.extend(then_checker.consumed_vars);
        self.consumed_vars.extend(else_checker.consumed_vars);
    }

    fn check_loop(&mut self, body: &[QLangCommand]) {
        let before_consumed = self.consumed_slots.clone();
        self.loop_depth += 1;
        for command in body {
            self.check_command(command);
        }
        self.loop_depth -= 1;
        let newly_consumed: Vec<_> = self
            .consumed_slots
            .difference(&before_consumed)
            .copied()
            .collect();
        for slot in newly_consumed {
            self.type_error(format!(
                "loop body consumes qubit {slot}; a later iteration could reuse a measured qubit"
            ));
        }
    }

    fn check_return(&mut self, expression: &Expression) {
        let actual = self.check_expression(expression);
        let Some(expected) = self.expected_return.clone() else {
            self.type_error("return may only be used inside a function".into());
            return;
        };
        if expected == QLangType::Void {
            self.type_error("a void function cannot return a value".into());
        } else if !types_compatible(&expected, &actual) {
            self.type_error(format!("function must return {expected}, found {actual}"));
        }
    }

    fn expect_type(&mut self, expression: &Expression, expected: &QLangType, context: &str) {
        let actual = self.check_expression(expression);
        if !types_compatible(expected, &actual) {
            self.type_error(format!("{context} expects {expected}, found {actual}"));
        }
    }

    fn check_qubit_argument(&mut self, expression: &Expression) {
        if let Expression::Variable(name) = expression {
            if self.consumed_vars.contains(name) {
                self.type_error(format!("qubit variable '{name}' has already been measured"));
            }
        }
        if let Some(slot) = self.resolve_qubit_slot(expression) {
            self.check_qubit_slot(slot, false);
        }
    }

    fn check_distinct_gate_qubits(
        &mut self,
        name: &str,
        args: &[Expression],
        parameter_types: &[QLangType],
    ) {
        let qubits: Vec<_> = args
            .iter()
            .zip(parameter_types)
            .filter_map(|(argument, parameter_type)| {
                (parameter_type == &QLangType::Qubit).then_some(argument)
            })
            .collect();
        let duplicate = qubits.iter().enumerate().any(|(index, left)| {
            qubits[index + 1..]
                .iter()
                .any(|right| self.same_qubit(left, right))
        });
        let reuses_control = self.qif_control.as_ref().is_some_and(|control| {
            qubits
                .iter()
                .any(|argument| self.same_qubit(control, argument))
        });
        if duplicate || reuses_control {
            self.errors
                .push(QLangError::DuplicateQubitArguments(name.to_string()));
        }
    }

    fn same_qubit(&self, left: &Expression, right: &Expression) -> bool {
        left == right
            || matches!(
                (self.resolve_qubit_slot(left), self.resolve_qubit_slot(right)),
                (Some(left), Some(right)) if left == right
            )
    }

    fn consume_qubit_argument(&mut self, expression: &Expression) {
        if let Expression::Variable(name) = expression {
            if !self.consumed_vars.insert(name.clone()) {
                self.type_error(format!("qubit variable '{name}' has already been measured"));
            }
        }
        if let Some(slot) = self.resolve_qubit_slot(expression) {
            self.check_qubit_slot(slot, true);
        }
    }

    fn resolve_qubit_slot(&self, expression: &Expression) -> Option<usize> {
        match expression {
            Expression::Number(number) if number.is_finite() && *number >= 0.0 => {
                Some(*number as usize)
            }
            Expression::Variable(name) => self.qubit_vars.get(name).copied(),
            Expression::Call(name, _) if name == "alloc" => extract_qubit_slot(expression),
            _ => None,
        }
    }

    fn check_qubit_slot(&mut self, slot: usize, consuming: bool) {
        if slot >= self.num_qubits {
            self.errors.push(QLangError::QubitOutOfRange(slot));
        } else if self.consumed_slots.contains(&slot) {
            self.errors.push(QLangError::QubitAlreadyMeasured(slot));
        } else if consuming {
            self.consumed_slots.insert(slot);
        }
    }

    pub fn infer_type(&self, expression: &Expression) -> QLangType {
        match expression {
            Expression::Number(number) if *number == number.floor() => QLangType::Int,
            Expression::Number(_) => QLangType::Float,
            Expression::Variable(name) => self.env.get(name).cloned().unwrap_or(QLangType::Unknown),
            Expression::Measure(_) => QLangType::Bit,
            Expression::Array(elements) => QLangType::Array(Box::new(
                elements
                    .first()
                    .map(|element| self.infer_type(element))
                    .unwrap_or(QLangType::Unknown),
            )),
            Expression::Index(array, _) => match self.infer_type(array) {
                QLangType::Array(element) => *element,
                _ => QLangType::Unknown,
            },
            Expression::BinaryOp { left, op, right } => match op {
                Operator::Eq
                | Operator::Neq
                | Operator::Lt
                | Operator::Gt
                | Operator::Le
                | Operator::Ge
                | Operator::And
                | Operator::Or => QLangType::Bool,
                _ if self.infer_type(left) == QLangType::Float
                    || self.infer_type(right) == QLangType::Float =>
                {
                    QLangType::Float
                }
                _ => QLangType::Int,
            },
            Expression::Call(name, _) if name == "alloc" => QLangType::Qubit,
            Expression::Call(name, _) => self
                .functions
                .get(name)
                .map(|signature| signature.return_type.clone())
                .unwrap_or(QLangType::Unknown),
        }
    }

    fn type_error(&mut self, message: String) {
        self.errors.push(QLangError::TypeError(message));
    }
}

fn guarantees_return(commands: &[QLangCommand]) -> bool {
    commands.iter().any(|command| match command {
        QLangCommand::Return(_) => true,
        QLangCommand::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => guarantees_return(then_branch) && guarantees_return(else_branch),
        _ => false,
    })
}

fn types_compatible(expected: &QLangType, actual: &QLangType) -> bool {
    expected == actual
        || expected == &QLangType::Unknown
        || actual == &QLangType::Unknown
        || (expected == &QLangType::Float && actual == &QLangType::Int)
}

fn gate_argument_compatible(expected: &QLangType, actual: &QLangType) -> bool {
    types_compatible(expected, actual)
        || (expected == &QLangType::Qubit && actual == &QLangType::Int)
}

fn is_numeric(value_type: &QLangType) -> bool {
    matches!(
        value_type,
        QLangType::Int | QLangType::Float | QLangType::Unknown
    )
}

fn is_condition_type(value_type: &QLangType) -> bool {
    matches!(
        value_type,
        QLangType::Bool | QLangType::Bit | QLangType::Int | QLangType::Float | QLangType::Unknown
    )
}

fn extract_qubit_slot(expression: &Expression) -> Option<usize> {
    match expression {
        Expression::Number(number) if number.is_finite() && *number >= 0.0 => {
            Some(*number as usize)
        }
        Expression::Call(name, args) if name == "alloc" => args.first().and_then(|argument| {
            if let Expression::Number(number) = argument {
                Some(*number as usize)
            } else {
                None
            }
        }),
        _ => None,
    }
}

fn is_qif_gate(name: &str) -> bool {
    matches!(
        name,
        "hadamard"
            | "paulix"
            | "pauliy"
            | "pauliz"
            | "s"
            | "sdagger"
            | "t"
            | "tdagger"
            | "rx"
            | "ry"
            | "rz"
            | "phase"
            | "u1"
            | "u2"
            | "u3"
            | "cnot"
            | "swap"
    )
}

fn native_gate_signature(name: &str) -> Option<Vec<QLangType>> {
    let qubits = match name {
        "hadamard" | "paulix" | "pauliy" | "pauliz" | "identity" | "s" | "sdagger" | "t"
        | "tdagger" => vec![QLangType::Qubit],
        "rx" | "ry" | "rz" | "phase" | "u1" => {
            vec![QLangType::Qubit, QLangType::Float]
        }
        "u2" => vec![QLangType::Qubit, QLangType::Float, QLangType::Float],
        "u3" => vec![
            QLangType::Qubit,
            QLangType::Float,
            QLangType::Float,
            QLangType::Float,
        ],
        "cnot" | "cy" | "cz" | "swap" | "iswap" => {
            vec![QLangType::Qubit, QLangType::Qubit]
        }
        "toffoli" | "fredkin" => vec![QLangType::Qubit, QLangType::Qubit, QLangType::Qubit],
        "controlled_u" | "cu" => vec![
            QLangType::Qubit,
            QLangType::Qubit,
            QLangType::Float,
            QLangType::Float,
            QLangType::Float,
            QLangType::Float,
        ],
        _ => return None,
    };
    Some(qubits)
}
