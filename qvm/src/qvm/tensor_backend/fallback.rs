// src/qvm/tensor_backend/fallback.rs

use super::TensorBackend;
use crate::types::qlang_complex::QLangComplex;
use ndarray::Array2;

impl TensorBackend {
    /// Executa uma versão fallback aplicando diretamente a matriz ao estado vetorial.
    /// Útil para debugging ou casos não suportados pela representação tensorial.
    pub fn execute_fallback(&mut self, matrix: &Array2<QLangComplex>, qubits: &[usize]) {
        println!(
            "Fallback ativado: aplicando matriz densa em {:?}",
            qubits
        );

        // 1. Converter TensorNetwork -> state vector
        let state = self.network.to_state_vector(); // precisa implementar isso

        // 2. Expand vetor para forma matricial se necessário
        let flat = ndarray::Array1::from(state);

        // 3. Aplicar a matriz no estado
        let result = matrix.dot(&flat);

        // 4. Atualizar: você pode salvar o vetor em self.network como full state
        self.network.overwrite_from_state_vector(result.to_vec());

        println!("Fallback concluído com vetor de estado atualizado.");
    }

}
