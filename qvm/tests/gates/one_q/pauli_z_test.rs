use qlang::types::qlang_complex::QLangComplex;
use qlang::gates::one_q::pauli_z::PauliZ;
use ndarray::array;

#[test]
fn test_pauli_z_matrix() {
    let z = PauliZ::new();
    let matrix = z.matrix;

    let expected = array![
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(-1.0, 0.0)]
    ];

    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
