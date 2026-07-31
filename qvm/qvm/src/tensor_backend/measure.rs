use super::TensorBackend;
use qlang_core::types::qlang_complex::QLangComplex;
use rand::Rng;

impl TensorBackend {
    pub fn measure(&mut self, qubit: usize) -> u8 {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        let mut state = self.network.to_state_vector();
        let probability_zero: f64 = state
            .iter()
            .enumerate()
            .filter(|(basis, _)| ((basis >> qubit) & 1) == 0)
            .map(|(_, amplitude)| amplitude.norm_sqr())
            .sum();
        let result = if rand::thread_rng().gen::<f64>() < probability_zero {
            0
        } else {
            1
        };
        for (basis, amplitude) in state.iter_mut().enumerate() {
            if ((basis >> qubit) & 1) != result {
                *amplitude = QLangComplex::zero();
            }
        }
        let norm = state.iter().map(QLangComplex::norm_sqr).sum::<f64>().sqrt();
        if norm > 0.0 {
            let divisor = QLangComplex::new(norm, 0.0);
            for amplitude in &mut state {
                *amplitude /= divisor;
            }
        }
        self.network.overwrite_from_state_vector(state);
        result as u8
    }

    pub fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> {
        qubits.iter().map(|qubit| self.measure(*qubit)).collect()
    }

    pub fn measure_all(&mut self) -> Vec<u8> {
        (0..self.num_qubits)
            .map(|qubit| self.measure(qubit))
            .collect()
    }

    pub fn finalize_measurements(&mut self) {}
}
