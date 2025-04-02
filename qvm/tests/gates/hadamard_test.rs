#[cfg(test)]
mod tests {
    use qlang::gates::hadamard::Hadamard;  // Importa do crate principal
    use qlang::qvm::cuda::types::CudaComplex;
    use ndarray::array;

    #[test]
    fn test_hadamard_matrix() {
        let h = Hadamard::new();
        let factor = 1.0 / (2.0_f64).sqrt();

        let expected = array![
            [CudaComplex::new(factor, 0.0), CudaComplex::new(factor, 0.0)],
            [CudaComplex::new(factor, 0.0), CudaComplex::new(-factor, 0.0)]
        ];

        assert_eq!(h.matrix, expected);
    }
}
