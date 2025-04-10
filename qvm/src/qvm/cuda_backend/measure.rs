// src/qvm/cuda_backend/measure.rs
#![cfg(feature = "cuda")]
use super::CudaBackend;
use crate::types::qlang_complex::QLangComplex;
use rand::Rng;
use cust::memory::CopyDestination;


impl CudaBackend {
    pub fn measure_all(&mut self) -> Vec<u8> {
        (0..self.num_qubits).map(|q| self.measure(q)).collect()
    }

    pub fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let r = self.measure(q);
            results.push(r);
        }

        results
    }

    pub fn measure(&mut self, qubit: usize) -> u8 {
        let dim = self.state.len();
        let mut host = vec![QLangComplex::default(); dim];
        self.state.copy_to(&mut host).unwrap();

        let prob_0: f64 = host
            .iter()
            .enumerate()
            .filter(|(i, _)| (i >> qubit) & 1 == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        let rand: f64 = rand::thread_rng().gen();
        let measured = if rand < prob_0 { 0 } else { 1 };

        for (i, amp) in host.iter_mut().enumerate() {
            if ((i >> qubit) & 1) != measured {
                *amp = QLangComplex::new(0.0, 0.0);
            }
        }

        let norm: f64 = host.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut host {
            *amp /= QLangComplex::new(norm, 0.0);
        }

        self.state.copy_from(&host).unwrap();
        measured as u8
    }
}
