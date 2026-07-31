use qlang_syntax::{
    ast::{Expression, QLangCommand},
    parser::{QLangLine, QLangParser},
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, Serialize)]
struct CircuitModel {
    qubits: usize,
    gates: Vec<CircuitGate>,
    diagnostics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CircuitGate {
    name: String,
    targets: Vec<usize>,
    parameters: Vec<f64>,
}

#[wasm_bindgen]
pub fn parse_circuit(source: &str) -> Result<JsValue, JsValue> {
    let model = parse_model(source).map_err(|errors| JsValue::from_str(&errors.join("\n")))?;
    serde_wasm_bindgen::to_value(&model).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn parse_model(source: &str) -> Result<CircuitModel, Vec<String>> {
    let mut parser = QLangParser::new();
    for line in source.lines() {
        parser.append(line);
    }
    parser.validate_lines();
    if parser.has_errors() {
        return Err(parser.get_errors().to_vec());
    }

    let mut model = CircuitModel {
        qubits: 0,
        gates: Vec::new(),
        diagnostics: Vec::new(),
    };
    for line in parser.get_commands() {
        if let QLangLine::Command(command) = line {
            project_command(command, &mut model);
        }
    }
    Ok(model)
}

fn project_command(command: &QLangCommand, model: &mut CircuitModel) {
    match command {
        QLangCommand::Create(qubits) => model.qubits = *qubits,
        QLangCommand::ApplyGate(name, arguments) => {
            let mut targets = Vec::new();
            let mut parameters = Vec::new();
            for (index, argument) in arguments.iter().enumerate() {
                match argument {
                    Expression::Number(value) if index < gate_qubit_arity(name) => {
                        targets.push(*value as usize)
                    }
                    Expression::Number(value) => parameters.push(*value),
                    _ => model.diagnostics.push(format!(
                        "viewer cannot resolve dynamic argument '{}' in gate '{}'",
                        argument, name
                    )),
                }
            }
            model.gates.push(CircuitGate {
                name: display_name(name).into(),
                targets,
                parameters,
            });
        }
        QLangCommand::Measure(qubit) => model.gates.push(CircuitGate {
            name: "measure".into(),
            targets: vec![*qubit],
            parameters: vec![],
        }),
        QLangCommand::MeasureMany(qubits) => {
            for qubit in qubits {
                model.gates.push(CircuitGate {
                    name: "measure".into(),
                    targets: vec![*qubit],
                    parameters: vec![],
                });
            }
        }
        QLangCommand::MeasureAll => {
            for qubit in 0..model.qubits {
                model.gates.push(CircuitGate {
                    name: "measure".into(),
                    targets: vec![qubit],
                    parameters: vec![],
                });
            }
        }
        QLangCommand::If {
            then_branch,
            else_branch,
            ..
        }
        | QLangCommand::QIf {
            then_branch,
            else_branch,
            ..
        } => {
            model.diagnostics.push(
                "conditional branches are displayed sequentially and are not simulated".into(),
            );
            for nested in then_branch {
                project_command(nested, model);
            }
            if let Some(branch) = else_branch {
                for nested in branch {
                    project_command(nested, model);
                }
            }
        }
        QLangCommand::While { body, .. }
        | QLangCommand::For { body, .. }
        | QLangCommand::FunctionDef { body, .. } => {
            model
                .diagnostics
                .push("nested body is shown once in the static circuit projection".into());
            for nested in body {
                project_command(nested, model);
            }
        }
        _ => {}
    }
}

fn gate_qubit_arity(name: &str) -> usize {
    match name {
        "cnot" | "cy" | "cz" | "swap" | "iswap" | "controlled_u" | "cu" => 2,
        "toffoli" | "fredkin" => 3,
        _ => 1,
    }
}

fn display_name(name: &str) -> &str {
    match name {
        "hadamard" => "h",
        "paulix" => "x",
        "pauliy" => "y",
        "pauliz" => "z",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_the_real_parser_ast() {
        let model = parse_model("create(2)\nh(0)\ncx(0, 1)\nm()\n").unwrap();
        assert_eq!(model.qubits, 2);
        assert_eq!(model.gates.len(), 4);
        assert_eq!(model.gates[1].name, "cnot");
        assert_eq!(model.gates[2].name, "measure");
    }

    #[test]
    fn returns_parser_diagnostics() {
        assert!(parse_model("h(").is_err());
    }
}
