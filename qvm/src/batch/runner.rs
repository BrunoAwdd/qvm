
use crate::qlang::QLang;
use crate::qvm::QVM;
use super::circuit_job::CircuitJob;
use rayon::prelude::*;

pub struct BatchRunner {
    pub jobs: Vec<CircuitJob>,
}

impl BatchRunner {
    pub fn new(jobs: Vec<CircuitJob>) -> Self {
        Self { jobs }
    }

    pub fn run_all(&self) -> Vec<QVM> {
        self.jobs
            .par_iter()
            .map(|job| {
                let mut qlang = QLang::new(job.num_qubits);
                job.commands.iter().cloned().for_each(|cmd| qlang.append(cmd));
                qlang.run();
                qlang.qvm.clone()
            })
            .collect()
    }

    pub fn from_files(paths: Vec<&str>) -> Self {
        let jobs = paths.into_iter().map(|path| {
            let content = std::fs::read_to_string(path).unwrap();
            let mut qlang = QLang::new(1); 
            qlang.run_from_str(&content);
            CircuitJob {
                num_qubits: qlang.qvm.num_qubits(),
                commands: qlang.ast.clone(),
            }
        }).collect();

        Self { jobs }
    }
}
