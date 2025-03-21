use qvm::gates::pauli_x::PauliX; // Importação correta do módulo
use num_complex::Complex;
use ndarray::array;

#[test]
fn test_pauli_x_matrix() {
    let x = PauliX::new();

    let expected = array![
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]
    ];

    assert_eq!(x.matrix, expected);
}
