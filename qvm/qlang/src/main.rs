use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use qlang::{
    ast::QLangCommand,
    parser::{QLangLine, QLangParser},
    type_checker::TypeChecker,
    QLang,
};

#[derive(Parser)]
#[command(name = "qlang", version, about = "Run and check QLang programs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse, type-check, and execute a QLang source file.
    Run {
        file: PathBuf,
        #[arg(short, long, default_value_t = 1)]
        qubits: usize,
    },
    /// Parse and type-check a QLang source file without executing it.
    Check {
        file: PathBuf,
        #[arg(short, long, default_value_t = 1)]
        qubits: usize,
    },
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn execute(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Run { file, qubits } => {
            let source = read_source(&file)?;
            let commands = parse_source(&source)?;
            let register_size = declared_qubits(&commands).unwrap_or(qubits);
            let mut qlang = QLang::new(register_size);
            qlang.ast = commands;
            qlang.try_run()
        }
        Command::Check { file, qubits } => {
            let source = read_source(&file)?;
            let commands = parse_source(&source)?;
            let register_size = declared_qubits(&commands).unwrap_or(qubits);
            let mut checker = TypeChecker::new(register_size);
            let errors = checker.check(&commands);
            if errors.is_empty() {
                println!("OK: {}", file.display());
                Ok(())
            } else {
                Err(errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }
}

fn read_source(file: &PathBuf) -> Result<String, String> {
    fs::read_to_string(file).map_err(|e| format!("failed to read '{}': {e}", file.display()))
}

fn parse_source(source: &str) -> Result<Vec<QLangCommand>, String> {
    let mut parser = QLangParser::new();
    for line in source.lines() {
        parser.append(line);
    }
    parser.validate_lines();
    if parser.has_errors() {
        return Err(parser.get_errors().join("\n"));
    }

    parser
        .get_commands()
        .iter()
        .map(|line| match line {
            QLangLine::Command(command) => Ok(command.clone()),
            QLangLine::Run | QLangLine::Reset => {
                Err("'run' and 'reset' are only supported in interactive mode".to_string())
            }
        })
        .collect()
}

fn declared_qubits(commands: &[QLangCommand]) -> Option<usize> {
    commands.iter().find_map(|command| match command {
        QLangCommand::Create(size) => Some(*size),
        _ => None,
    })
}
