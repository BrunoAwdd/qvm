pub mod qlang;
pub mod qvm;
pub mod gates;
pub mod state;
pub mod batch;
pub mod types;

use std::time::Instant;

use crate::{
    qlang::QLang, 
    batch::{circuit_job::CircuitJob, runner::BatchRunner}};


fn main() {
    let start = Instant::now();
   


    let mut jobs = vec![];
    let mut qlangs = vec![]; // Armazena os QLangs para impressão posterior

    for i in 1..=4 {
        let path = format!("../batch/batch_test_{}.ql", i);
        let mut qlang = QLang::new(1); // valor dummy, será sobrescrito
        qlang.run_qlang_from_file(&path);

        let num_qubits = 3;

        let job = CircuitJob {
            num_qubits,
            commands: qlang.ast.clone(),
        };

        jobs.push(job);
        qlangs.push(qlang); // guarda o QLang
    }

    let batch = BatchRunner::new(jobs);
    let results = batch.run_all();

    for (i, (qvm, qlang)) in results.iter().zip(qlangs.iter()).enumerate() {
        println!("=== Job {} ===", i + 1);
        println!("Código QLang:");
        println!("{}", qlang.to_source()); 
        println!("Resultado:");
        qvm.display();
        println!();
    }

    let duration = start.elapsed();

    println!("⏱️ Tempo total de execução: {:.3?}", duration);

    for i in 1..=4 {
        let path = format!("../batch/batch_test_{}.ql", i);
        let mut qlang = QLang::new(1);
        qlang.run_qlang_from_file(&path);
    }

    let duration2 = start.elapsed();

    println!("⏱️ Tempo total de execução: {:.3?} - {:.3?}", duration, duration2-duration);
}
