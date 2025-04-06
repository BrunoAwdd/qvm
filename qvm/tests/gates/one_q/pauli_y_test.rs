use qlang::gates::one_q::pauli_y::PauliY;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_pauli_y_matrix() {
    let y = PauliY::new();

    let expected = array![
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, -1.0)],
        [QLangComplex::new(0.0, 1.0), QLangComplex::new(0.0, 0.0)]
    ];

    assert_eq!(y.matrix, expected);
}
