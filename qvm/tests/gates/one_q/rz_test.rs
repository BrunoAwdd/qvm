use qlang::gates::rotation_q::rz::RZ;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_rz_pi() {
    let rz = RZ::new(PI);
    let matrix = rz.matrix;

    let zero = QLangComplex::zero();
    let i = QLangComplex::i();
    let neg_i = QLangComplex::neg_i();

    let expected = array![
        [neg_i, zero],
        [zero, i],
    ];
    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
