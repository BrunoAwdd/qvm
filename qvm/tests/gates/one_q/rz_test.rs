use qlang::gates::one_q::rz::RZ;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_rz_pi() {
    let rz = RZ::new(PI);
    let matrix = rz.matrix;
    let expected = array![
        [QLangComplex::new(0.0, -1.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 1.0)],
    ];
    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
