use qvm::gates::pauli_z::PauliZ;
use num_complex::Complex;
use ndarray::array;

#[test]
fn test_pauli_z_matrix() {
    let z = PauliZ::new();

    let expected = array![
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)]
    ];

    assert_eq!(z.matrix, expected);
}
