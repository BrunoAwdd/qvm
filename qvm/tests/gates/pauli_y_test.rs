use qvm::gates::pauli_y::PauliY;
use num_complex::Complex;
use ndarray::array;

#[test]
fn test_pauli_y_matrix() {
    let y = PauliY::new();

    let expected = array![
        [Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
        [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]
    ];

    assert_eq!(y.matrix, expected);
}
