pub mod qlang;
pub mod qvm;
pub mod gates;
pub mod state;

use std::env;  // Para acessar argumentos passados na linha de comando
use crate::qlang::QLang;  // Importando a QLang

fn main() {
    // Recebe os argumentos passados na linha de comando
    let args: Vec<String> = env::args().collect();

    // Verifica se o caminho do arquivo foi fornecido
    if args.len() < 2 {
        println!("Erro: Por favor, forneça o caminho para um arquivo .ql.");
        return;
    }

    // O primeiro argumento (args[1]) será o caminho do arquivo
    let file_path = &args[1];

    // Inicializa o interpretador QLang com 2 qubits
    let mut qlang = QLang::new(2);

    // Executa o script QLang fornecido
    qlang.run_qlang_from_file(file_path);
}
