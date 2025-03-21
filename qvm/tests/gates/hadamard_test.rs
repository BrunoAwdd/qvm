#[cfg(test)]
mod tests {
    use qvm::gates::hadamard::Hadamard;  // Importa do crate principal
    use num_complex::Complex;
    use ndarray::array;

    #[test]
    fn test_hadamard_matrix() {
        let h = Hadamard::new();
        let factor = 1.0 / (2.0_f64).sqrt();

        let expected = array![
            [Complex::new(factor, 0.0), Complex::new(factor, 0.0)],
            [Complex::new(factor, 0.0), Complex::new(-factor, 0.0)]
        ];

        assert_eq!(h.matrix, expected);
    }
}
