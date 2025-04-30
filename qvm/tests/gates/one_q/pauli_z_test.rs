use qlang::types::qlang_complex::QLangComplex;
use qlang::gates::one_q::pauli_z::PauliZ;
use ndarray::array;

#[test]
fn test_pauli_z_matrix() {
    let z = PauliZ::new();
    let matrix = z.matrix;

    let zero = QLangComplex::zero();
    let one = QLangComplex::one();
    let neg_one = QLangComplex::neg_one();

    let expected = array![
        [one, zero],
        [zero, neg_one]
    ];

    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
