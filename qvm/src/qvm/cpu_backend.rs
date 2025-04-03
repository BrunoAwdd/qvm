use ndarray::Array1;
use rand::thread_rng;
use rand::Rng;

use crate::state::quantum_state::QuantumState;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use crate::qvm::backend::QuantumBackend;
use crate::qvm::cuda::types::CudaComplex;

pub struct CpuBackend {
    state: QuantumState,
}

impl CpuBackend {
    pub fn new(num_qubits: usize) -> Self {
        let state = QuantumState::new(num_qubits);
        Self { state }
    }
    
}

impl QuantumBackend for CpuBackend {
    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        let dim = self.state.state_vector.len();
        let mut new_state = Array1::<CudaComplex>::zeros(dim);
        let gate_matrix = gate.matrix();

        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                let j = i ^ (1 << qubit);
                let a = self.state.state_vector[i];
                let b = self.state.state_vector[j];

                new_state[i] = gate_matrix[[0, 0]] * a + gate_matrix[[0, 1]] * b;
                new_state[j] = gate_matrix[[1, 0]] * a + gate_matrix[[1, 1]] * b;
            }
        }


        self.state.state_vector = new_state;
    }

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize) {
        let n = self.state.num_qubits;
        assert!(
            q1 < self.num_qubits() && q2 < self.num_qubits() && q1 != q2,
            "Gate de 2 qubits requer índices distintos e válidos (0..{}). Recebido: {}, {}",
            self.num_qubits(), q1, q2
        );

        let dim = 1 << n;
        let mut new_state = Array1::<CudaComplex>::zeros(dim);
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
                new_state[j] += amp * self.state.state_vector[i];
            }
        }

        self.state.state_vector = new_state;
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        let n = self.num_qubits();
        assert!(q0 < n && q1 < n && q2 < n, "Qubit fora do range");
        assert!(q0 != q1 && q0 != q2 && q1 != q2, "Qubits devem ser distintos");

        let indices = [q0, q1, q2];
        let mut sorted = indices;
        sorted.sort_unstable();

        let permute_to = [q0, q1, q2];
        let gate_matrix = gate.matrix(); // 8x8

        let mut new_state = self.state.clone();

        let size = 1 << n;

        for i in 0..size {
            let b0 = (i >> q0) & 1;
            let b1 = (i >> q1) & 1;
            let b2 = (i >> q2) & 1;

            let input_idx = (b0 << 2) | (b1 << 1) | b2;

            for output in 0..8 {
                let out_b0 = (output >> 2) & 1;
                let out_b1 = (output >> 1) & 1;
                let out_b2 = (output >> 0) & 1;

                let mut j = i;
                j = (j & !(1 << q0)) | (out_b0 << q0);
                j = (j & !(1 << q1)) | (out_b1 << q1);
                j = (j & !(1 << q2)) | (out_b2 << q2);

                let amp = gate_matrix[[output, input_idx]] * self.state.state_vector[i];
                new_state.state_vector[j] = new_state.state_vector[j] + amp;
            }
        }

        self.state = new_state;
    }
    

    fn measure(&mut self, qubit: usize) -> u8 {
        let dim = self.state.state_vector.len();
        let mut prob_0 = 0.0;
        
        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                prob_0 += self.state.state_vector[i].norm_sqr();
            }
        }

        let mut rng = thread_rng();
        let rand_value: f64 = rng.gen();

        let measured_value = if rand_value < prob_0 { 0 } else { 1 };

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit != measured_value {
                self.state.state_vector[i] = CudaComplex::new(0.0, 0.0);
            }
        }

        let norm_factor = (self.state.state_vector.iter().map(|x| x.norm_sqr()).sum::<f64>()).sqrt();
        let norm_complex = CudaComplex::new(norm_factor, 0.0);
        if norm_factor != 0.0 {
            self.state.state_vector.mapv_inplace(|x| {
                let mut c = x;
                c /= norm_complex;
                c
            });
        }

        measured_value as u8
    }

    fn measure_all(&mut self) -> Vec<u8> {
        let mut results = vec![0; self.state.num_qubits];
        for qubit in 0..self.state.num_qubits {
            results[qubit] = self.measure(qubit);
        }
        results
    }

    fn display(&self) {
        println!("{}", self.state);
    }

    fn reset(&mut self, num_qubits: usize) {
        if self.state.num_qubits != num_qubits {
            self.state = QuantumState::new(num_qubits);
        } else {
            self.state.reset_state();
        }
    }

    fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    fn state_vector(&self) -> Vec<CudaComplex> {
        self.state.state_vector.iter().map(|c| CudaComplex::new(c.re, c.im)).collect()
    }

    fn box_clone(&self) -> Box<dyn QuantumBackend> {
        Box::new(self.clone())
    }
    
    fn name(&self) -> &'static str {
        "CPU"
    }

}


impl Clone for CpuBackend {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

pub type Backend = CpuBackend; 