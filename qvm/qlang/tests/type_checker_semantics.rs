use qlang::{
    ast::{Expression, QLangCommand, TypeAnnotation},
    type_checker::TypeChecker,
};

fn number(value: f64) -> Expression {
    Expression::Number(value)
}

fn variable(name: &str) -> Expression {
    Expression::Variable(name.into())
}

fn alloc(slot: usize) -> Expression {
    Expression::Call("alloc".into(), vec![number(slot as f64)])
}

fn messages(commands: &[QLangCommand]) -> Vec<String> {
    TypeChecker::new(4)
        .check(commands)
        .into_iter()
        .map(|error| error.to_string())
        .collect()
}

#[test]
fn accepts_a_well_typed_function_and_call() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "rotate".into(),
            params: vec!["q".into(), "angle".into()],
            param_types: vec![Some(TypeAnnotation::Qubit), Some(TypeAnnotation::Float)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::ApplyGate(
                "rx".into(),
                vec![variable("q"), variable("angle")],
            )],
        },
        QLangCommand::ApplyGate("rotate".into(), vec![number(0.0), number(0.5)]),
    ];

    assert!(messages(&commands).is_empty());
}

#[test]
fn rejects_assignment_that_changes_variable_type() {
    let commands = vec![
        QLangCommand::Let {
            name: "value".into(),
            type_ann: Some(TypeAnnotation::Int),
            value: number(1.0),
        },
        QLangCommand::Assign {
            name: "value".into(),
            value: number(1.5),
        },
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("cannot assign")));
}

#[test]
fn rejects_heterogeneous_arrays() {
    let commands = vec![QLangCommand::Let {
        name: "values".into(),
        type_ann: None,
        value: Expression::Array(vec![number(1.0), number(1.5)]),
    }];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("array elements")));
}

#[test]
fn checks_function_argument_types() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "takes_int".into(),
            params: vec!["value".into()],
            param_types: vec![Some(TypeAnnotation::Int)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![],
        },
        QLangCommand::ApplyGate("takes_int".into(), vec![number(1.5)]),
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("expects int")));
}

#[test]
fn checks_declared_return_type() {
    let commands = vec![QLangCommand::FunctionDef {
        name: "answer".into(),
        params: vec![],
        param_types: vec![],
        return_type: Some(TypeAnnotation::Int),
        body: vec![QLangCommand::Return(number(1.5))],
    }];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("must return int")));
}

#[test]
fn requires_non_void_functions_to_return_on_every_path() {
    let commands = vec![QLangCommand::FunctionDef {
        name: "answer".into(),
        params: vec!["condition".into()],
        param_types: vec![Some(TypeAnnotation::Bool)],
        return_type: Some(TypeAnnotation::Int),
        body: vec![QLangCommand::If {
            condition: variable("condition"),
            then_branch: vec![QLangCommand::Return(number(1.0))],
            else_branch: None,
        }],
    }];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("does not return on every path")));
}

#[test]
fn rejects_break_outside_a_loop() {
    assert!(messages(&[QLangCommand::Break])
        .iter()
        .any(|error| error.contains("inside a loop")));
}

#[test]
fn prevents_two_variables_from_owning_the_same_qubit() {
    let commands = vec![
        QLangCommand::Create(2),
        QLangCommand::Let {
            name: "first".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::Let {
            name: "second".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("already owned")));
}

#[test]
fn moving_a_qubit_invalidates_the_source_variable() {
    let commands = vec![
        QLangCommand::Create(2),
        QLangCommand::Let {
            name: "source".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::Let {
            name: "target".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: variable("source"),
        },
        QLangCommand::ApplyGate("hadamard".into(), vec![variable("source")]),
    ];

    let errors = messages(&commands);
    assert!(
        errors.iter().any(|error| error.contains("has been moved")),
        "{errors:?}"
    );
}

#[test]
fn propagates_measurement_effects_from_function_calls() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "observe".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::Let {
                name: "result".into(),
                type_ann: Some(TypeAnnotation::Bit),
                value: Expression::Measure(Box::new(variable("q"))),
            }],
        },
        QLangCommand::Let {
            name: "source".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::ApplyGate("observe".into(), vec![variable("source")]),
        QLangCommand::ApplyGate("hadamard".into(), vec![variable("source")]),
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("already been measured")));
}

#[test]
fn propagates_measurement_effects_through_forward_function_calls() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "outer".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::ApplyGate("inner".into(), vec![variable("q")])],
        },
        QLangCommand::FunctionDef {
            name: "inner".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::Let {
                name: "result".into(),
                type_ann: Some(TypeAnnotation::Bit),
                value: Expression::Measure(Box::new(variable("q"))),
            }],
        },
        QLangCommand::Let {
            name: "source".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::ApplyGate("outer".into(), vec![variable("source")]),
        QLangCommand::ApplyGate("hadamard".into(), vec![variable("source")]),
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("already been measured")));
}

#[test]
fn propagates_move_effects_from_function_calls() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "take".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::Let {
                name: "owned".into(),
                type_ann: Some(TypeAnnotation::Qubit),
                value: variable("q"),
            }],
        },
        QLangCommand::Let {
            name: "source".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::ApplyGate("take".into(), vec![variable("source")]),
        QLangCommand::ApplyGate("hadamard".into(), vec![variable("source")]),
    ];

    let errors = messages(&commands);
    assert!(
        errors.iter().any(|error| error.contains("has been moved")),
        "{errors:?}"
    );
}

#[test]
fn unitary_function_parameters_are_borrowed() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "rotate".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::ApplyGate(
                "hadamard".into(),
                vec![variable("q")],
            )],
        },
        QLangCommand::Let {
            name: "source".into(),
            type_ann: Some(TypeAnnotation::Qubit),
            value: alloc(0),
        },
        QLangCommand::ApplyGate("rotate".into(), vec![variable("source")]),
        QLangCommand::ApplyGate("paulix".into(), vec![variable("source")]),
    ];

    assert!(messages(&commands).is_empty());
}

#[test]
fn merges_consumed_qubits_from_conditional_branches() {
    let commands = vec![
        QLangCommand::Create(2),
        QLangCommand::If {
            condition: number(1.0),
            then_branch: vec![QLangCommand::Measure(0)],
            else_branch: Some(vec![]),
        },
        QLangCommand::ApplyGate("hadamard".into(), vec![number(0.0)]),
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("already been measured")));
}

#[test]
fn rejects_consuming_a_qubit_in_a_repeating_loop() {
    let commands = vec![
        QLangCommand::Create(2),
        QLangCommand::While {
            condition: number(1.0),
            body: vec![QLangCommand::Measure(0)],
        },
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("later iteration")));
}

#[test]
fn reports_unknown_variables() {
    let commands = vec![QLangCommand::ApplyGate(
        "hadamard".into(),
        vec![variable("missing")],
    )];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("unknown variable")));
}

#[test]
fn rejects_non_unitary_commands_inside_qif() {
    let commands = vec![QLangCommand::QIf {
        condition: number(0.0),
        then_branch: vec![QLangCommand::Measure(1)],
        else_branch: Some(vec![QLangCommand::Let {
            name: "value".into(),
            type_ann: None,
            value: number(1.0),
        }]),
    }];

    let errors = messages(&commands);
    assert_eq!(
        errors
            .iter()
            .filter(|error| error.contains("inside qif"))
            .count(),
        2
    );
}

#[test]
fn rejects_user_functions_inside_qif_until_they_have_unitary_effects() {
    let commands = vec![
        QLangCommand::FunctionDef {
            name: "flip".into(),
            params: vec!["q".into()],
            param_types: vec![Some(TypeAnnotation::Qubit)],
            return_type: Some(TypeAnnotation::Void),
            body: vec![QLangCommand::ApplyGate(
                "paulix".into(),
                vec![variable("q")],
            )],
        },
        QLangCommand::QIf {
            condition: number(0.0),
            then_branch: vec![QLangCommand::ApplyGate("flip".into(), vec![number(1.0)])],
            else_branch: None,
        },
    ];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("inside qif")));
}

#[test]
fn accepts_a_supported_unitary_gate_inside_qif() {
    let commands = vec![QLangCommand::QIf {
        condition: number(0.0),
        then_branch: vec![QLangCommand::ApplyGate(
            "hadamard".into(),
            vec![number(1.0)],
        )],
        else_branch: None,
    }];

    assert!(messages(&commands).is_empty());
}

#[test]
fn rejects_duplicate_qubits_in_multi_qubit_gates() {
    let commands = vec![
        QLangCommand::ApplyGate("cnot".into(), vec![number(0.0), number(0.0)]),
        QLangCommand::ApplyGate(
            "toffoli".into(),
            vec![number(0.0), number(1.0), number(1.0)],
        ),
    ];

    assert_eq!(
        messages(&commands)
            .iter()
            .filter(|error| error.contains("distinct qubit"))
            .count(),
        2
    );
}

#[test]
fn rejects_a_qif_gate_that_reuses_its_control() {
    let commands = vec![QLangCommand::QIf {
        condition: number(0.0),
        then_branch: vec![QLangCommand::ApplyGate("paulix".into(), vec![number(0.0)])],
        else_branch: None,
    }];

    assert!(messages(&commands)
        .iter()
        .any(|error| error.contains("distinct qubit")));
}
