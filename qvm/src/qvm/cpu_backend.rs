use ndarray::Array1;
use rand::{thread_rng, Rng};

use crate::{
    gates::quantum_gate_abstract::QuantumGateAbstract,
    qvm::backend::QuantumBackend,
    state::quantum_state::QuantumState,
    types::qlang_complex::QLangComplex,
};

pub struct CpuBackend {
    state: QuantumState,
}

impl CpuBackend {
    pub fn new(num_qubits: usize) -> Self {
        Self { state: QuantumState::new(num_qubits) }
    }
}

impl QuantumBackend for CpuBackend {
    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        let dim = self.state.state_vector.len();
        let mut new_state = Array1::zeros(dim);
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

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        let n = self.num_qubits();
        assert_valid_qubits("2Q Gate", n, &[q0, q1]);

        let dim = 1 << n;
        let mut new_state = Array1::zeros(dim);
        let matrix = gate.matrix();

        for i in 0..dim {
            let (high, low) = if q0 > q1 { (q0, q1) } else { (q1, q0) };
            let b_high = (i >> high) & 1;
            let b_low  = (i >> low) & 1;
            let input = (b_high << 1) | b_low;

            for output in 0..4 {
                let o_high = (output >> 1) & 1;
                let o_low  = output & 1;

                let mut j = i;
                j = (j & !(1 << high)) | (o_high << high);
                j = (j & !(1 << low))  | (o_low  << low);

                new_state[j] += matrix[[output, input]] * self.state.state_vector[i];
            }
        }

        self.state.state_vector = new_state;
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        let n = self.num_qubits();
        assert_valid_qubits("3Q Gate", n, &[q0, q1, q2]);

        let dim = 1 << n;
        let matrix = gate.matrix();
        let mut new_state = self.state.clone();

        for i in 0..dim {
            let input = ((i >> q0) & 1) << 2 | ((i >> q1) & 1) << 1 | ((i >> q2) & 1);

            for output in 0..8 {
                let bits = [(output >> 2) & 1, (output >> 1) & 1, output & 1];

                let mut j = i;
                j = (j & !(1 << q0)) | (bits[0] << q0);
                j = (j & !(1 << q1)) | (bits[1] << q1);
                j = (j & !(1 << q2)) | (bits[2] << q2);

                new_state.state_vector[j] += matrix[[output, input]] * self.state.state_vector[i];
            }
        }

        self.state = new_state;
    }

    fn measure(&mut self, qubit: usize) -> u8 {
        let prob_0: f64 = self.state.state_vector
            .iter()
            .enumerate()
            .filter(|(i, _)| ((i >> qubit) & 1) == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        let outcome = if thread_rng().gen::<f64>() < prob_0 { 0 } else { 1 };

        for (i, amp) in self.state.state_vector.iter_mut().enumerate() {
            if ((i >> qubit) & 1) != outcome {
                *amp = QLangComplex::new(0.0, 0.0);
            }
        }

        let norm: f64 = self.state.state_vector.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm != 0.0 {
            let norm_complex = QLangComplex::new(norm, 0.0);
            for amp in self.state.state_vector.iter_mut() {
                *amp /= norm_complex;
            }
        }

        outcome as u8
    }

    fn measure_all(&mut self) -> Vec<u8> {
        (0..self.state.num_qubits)
            .map(|q| self.measure(q))
            .collect()
    }

    fn display(&self) {
        println!("State ({} qubits):", self.num_qubits());
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

    fn state_vector(&self) -> Vec<QLangComplex> {
        self.state.state_vector.to_vec()
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
        Self { state: self.state.clone() }
    }
}

pub type Backend = CpuBackend;

fn assert_valid_qubits(label: &str, total: usize, indices: &[usize]) {
    for &q in indices {
        assert!(q < total, "{}: qubit {} out of bounds (max = {})", label, q, total - 1);
    }
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            assert!(indices[i] != indices[j], "{}: qubits must be distinct", label);
        }
    }
}
