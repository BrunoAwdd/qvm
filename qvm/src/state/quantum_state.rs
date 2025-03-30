use ndarray::{Array1};
use num_complex::Complex;
use rand::prelude::*;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract; // Importar o trait corretamente
use std::fmt;

#[derive(Debug, Clone)] 
pub struct QuantumState {
    pub num_qubits: usize,
    pub state_vector: Array1<Complex<f64>>,
}

impl fmt::Display for QuantumState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantum State with {} qubits:", self.num_qubits)?;
        for (i, amp) in self.state_vector.iter().enumerate() {
            if amp.norm_sqr() > 1e-10 {
                writeln!(f, "|{:0width$b}⟩: {:.4} + {:.4}i", i, amp.re, amp.im, width = self.num_qubits)?;
            }
        }
        Ok(())
    }
}

impl QuantumState {
    pub fn new(num_qubits: usize) -> Self {
        let dim: usize = 1 << num_qubits;
        let mut state_vector = Array1::<Complex<f64>>::zeros(dim);
        state_vector[0] = Complex::new(1.0, 0.0);
        Self { num_qubits, state_vector }
    }

    pub fn apply_gate<T: QuantumGateAbstract>(&mut self, gate: &T, qubit: usize) {
        let dim = self.state_vector.len();
        let mut new_state = Array1::<Complex<f64>>::zeros(dim);
        let gate_matrix = gate.matrix();

        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                let j = i ^ (1 << qubit);
                let a = self.state_vector[i];
                let b = self.state_vector[j];

                new_state[i] = gate_matrix[[0, 0]] * a + gate_matrix[[0, 1]] * b;
                new_state[j] = gate_matrix[[1, 0]] * a + gate_matrix[[1, 1]] * b;
            }
        }

        self.state_vector = new_state;
    }

    pub fn apply_gate_2q<T: QuantumGateAbstract>(&mut self, gate: &T, q1: usize, q2: usize) {
        let n = self.num_qubits;
        assert!(q1 < n && q2 < n && q1 != q2);

        let dim = 1 << n;
        let mut new_state = Array1::<Complex<f64>>::zeros(dim);
        let gate_matrix = gate.matrix();

        for i in 0..dim {
            let b1 = (i >> q1) & 1;
            let b2 = (i >> q2) & 1;
            let input_index = (b1 << 1) | b2;

            for output_index in 0..4 {
                let o1 = (output_index >> 1) & 1;
                let o2 = output_index & 1;

                let mut j = i;
                j &= !(1 << q1);
                j &= !(1 << q2);
                j |= o1 << q1;
                j |= o2 << q2;

                let amp = gate_matrix[[output_index, input_index]];
                new_state[j] += amp * self.state_vector[i];
            }
        }

        self.state_vector = new_state;
    }

    pub fn measure(&mut self, qubit: usize) -> u8 {
        let dim = self.state_vector.len();
        let mut prob_0 = 0.0;
        
        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                prob_0 += self.state_vector[i].norm_sqr();
            }
        }

        let mut rng = thread_rng();
        let rand_value: f64 = rng.gen();

        let measured_value = if rand_value < prob_0 { 0 } else { 1 };

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit != measured_value {
                self.state_vector[i] = Complex::new(0.0, 0.0);
            }
        }

        let norm_factor = (self.state_vector.iter().map(|x| x.norm_sqr()).sum::<f64>()).sqrt();
        let norm_complex = Complex::new(norm_factor, 0.0);
        if norm_factor != 0.0 {
            self.state_vector /= norm_complex;
        }

        measured_value as u8
    }

    pub fn measure_all(&mut self) -> Vec<u8> {
        let mut results = vec![0; self.num_qubits];
        for qubit in 0..self.num_qubits {
            results[qubit] = self.measure(qubit);
        }
        results
    }

    pub fn display(&self) {
        println!("Quantum State with {} qubits:", self.num_qubits);
        for (i, amp) in self.state_vector.iter().enumerate() {
            if amp.norm_sqr() > 1e-10 {
                println!("|{}⟩: {:.4} + {:.4}i", i, amp.re, amp.im);
            }
        }
    }
}