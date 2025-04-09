use qlang::gates::rotation_q::ry::RY;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_ry_pi() {
    let ry = RY::new(PI);
    let matrix = ry.matrix;
    let expected = array![
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(-1.0, 0.0)],
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
    ];
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
}
