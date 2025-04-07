// src/qvm/cuda_backend/fallback.rs
#![cfg(feature = "cuda")]
use super::CudaBackend;
use crate::types::qlang_complex::QLangComplex;
use ndarray::Array2;

use cust::memory::CopyDestination;


impl CudaBackend {
    pub fn execute_fallback(
        &mut self,
        matrix: &Array2<QLangComplex>,
        qubits: &[usize],
    ) {
        let dim = 1 << self.num_qubits;
        let mut host = vec![QLangComplex::default(); dim];
        self.state.copy_to(&mut host).expect("Erro ao copiar do device");
        let mut new_state = host.clone();

        match qubits.len() {
            1 => {
                let q = qubits[0];
                for i in 0..dim {
                    if (i >> q) & 1 == 0 {
                        let j = i ^ (1 << q);
                        let a = host[i];
                        let b = host[j];
                        new_state[i] = matrix[(0, 0)] * a + matrix[(0, 1)] * b;
                        new_state[j] = matrix[(1, 0)] * a + matrix[(1, 1)] * b;
                    }
                }
            }
            2 => {
                let (q1, q2) = (qubits[0], qubits[1]);
                for i in 0..dim {
                    let b1 = (i >> q1) & 1;
                    let b2 = (i >> q2) & 1;
                    let input_idx = (b1 << 1) | b2;

                    for out_idx in 0..4 {
                        let o1 = (out_idx >> 1) & 1;
                        let o2 = out_idx & 1;

                        let mut j = i & !(1 << q1) & !(1 << q2);
                        j |= o1 << q1;
                        j |= o2 << q2;

                        new_state[j] += matrix[(out_idx, input_idx)] * host[i];
                    }
                }
            }
            3 => {
                let (q0, q1, q2) = (qubits[0], qubits[1], qubits[2]);
                for i in 0..dim {
                    let b0 = (i >> q0) & 1;
                    let b1 = (i >> q1) & 1;
                    let b2 = (i >> q2) & 1;
                    let input_idx = (b0 << 2) | (b1 << 1) | b2;

                    for out_idx in 0..8 {
                        let o0 = (out_idx >> 2) & 1;
                        let o1 = (out_idx >> 1) & 1;
                        let o2 = out_idx & 1;

                        let mut j = i & !(1 << q0) & !(1 << q1) & !(1 << q2);
                        j |= o0 << q0;
                        j |= o1 << q1;
                        j |= o2 << q2;

                        new_state[j] += matrix[(out_idx, input_idx)] * host[i];
                    }
                }
            }
            _ => panic!("Fallback suportado apenas para até 3 qubits."),
        }

        self.state.copy_from(&new_state).expect("Erro ao copiar para o device");
    }
}
