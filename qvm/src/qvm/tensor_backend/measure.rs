// src/qvm/tensor_backend/measure.rs

use rand::Rng;
use crate::types::qlang_complex::QLangComplex;
use super::TensorBackend;

impl TensorBackend {
    fn _measure(&mut self, q: usize) -> u8 {
        let tensor = &mut self.network.nodes[q].tensor;
        let shape = tensor.shape().to_vec(); 
        println!("🔍 Medindo qubit {}: shape = {:?}", q, shape);

        let slice_0 = tensor.index_axis(ndarray::Axis(1), 0);
        let slice_1 = tensor.index_axis(ndarray::Axis(1), 1);

        let norm_0 = slice_0.iter().map(|c| c.norm_sqr()).sum::<f64>();
        let norm_1 = slice_1.iter().map(|c| c.norm_sqr()).sum::<f64>();
        let total = norm_0 + norm_1;
        let p0 = norm_0 / total;

        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let result = if r < p0 { 0 } else { 1 };

        let norm_factor = if result == 0 {
            norm_0.sqrt().max(1e-12)
        } else {
            norm_1.sqrt().max(1e-12)
        };

        for i in 0..shape[0] {
            for j in 0..2 {
                for k in 0..shape[2] {
                    if j == result {
                        tensor[[i, j, k]] = tensor[[i, j, k]] / norm_factor;
                    } else {
                        tensor[[i, j, k]] = QLangComplex::default();
                    }
                }
            }
        }

        result as u8
    }

    pub fn measure(&mut self, q: usize) -> u8 {
        println!("MEDIÇÃO NO QUIBIT {}", q);
        let result = self._measure(q);
        let new_state = self.network.to_state_vector();
        self.network.overwrite_from_state_vector(new_state);

        result
    }

    pub fn measure_many(&mut self, qs: &Vec<usize>) -> Vec<u8> {
        let mut results = Vec::with_capacity(qs.len());

        println!("MEDIÇÃO DE VÁRIOS QUIBITS");

        for &q in qs {
            let r = self._measure(q);
            results.push(r);
            println!("MEDIÇÃO NO QUIBIT {}: VALOR {}", q, r);
        }

        // Após cada medição, atualiza a rede com o novo estado colapsado
        let new_state = self.network.to_state_vector();
        self.network.overwrite_from_state_vector(new_state);

        results
    }



    pub fn measure_all(&mut self) -> Vec<u8> {
        let results: Vec<u8> = (0..self.num_qubits)
            .map(|q| self._measure(q))
            .collect();

        // Agora sim, reescreve a rede inteira com base no novo vetor de estado
        let new_state = self.network.to_state_vector();
        self.network.overwrite_from_state_vector(new_state);

        results
    }


    pub fn finalize_measurements(&mut self) {
        let new_state = self.network.to_state_vector();
        self.network.overwrite_from_state_vector(new_state);
    }
}
