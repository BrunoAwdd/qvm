use qlang::gates::one_q::identity::Identity;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_identity_gate_matrix() {
    let id = Identity::new();
    let matrix = id.matrix;

    let expected = array![
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0)],
    ];

    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
