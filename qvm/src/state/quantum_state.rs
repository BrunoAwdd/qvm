
use ndarray::Array1;
use std::fmt;
use crate::qvm::cuda::types::CudaComplex;

#[derive(Debug, Clone)] 

pub struct QuantumState {
    pub num_qubits: usize,
    pub state_vector: Array1<CudaComplex>,
}

impl fmt::Display for QuantumState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantum State with {} qubits:", self.num_qubits)?;
        for (i, amp) in self.state_vector.iter().enumerate() {
            if amp.norm_sqr() > 1e-10 {
                writeln!(
                    f,
                    "|{:0width$b}⟩: {:.4} + {:.4}i",
                    i,
                    amp.re,
                    amp.im,
                    width = self.num_qubits
                )?;
            }
        }
        Ok(())
    }
}

impl QuantumState {
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut state_vector = Array1::<CudaComplex>::zeros(dim);
        state_vector[0] = CudaComplex::new(1.0, 0.0);
        Self { num_qubits, state_vector }
    }

    pub fn reset_state(&mut self) {
        self.state_vector.fill(CudaComplex::new(0.0, 0.0));
        self.state_vector[0] = CudaComplex::new(1.0, 0.0);
    }
}
