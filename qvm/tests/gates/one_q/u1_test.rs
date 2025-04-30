use qlang::gates::rotation_q::u1::U1;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;

#[test]
fn test_u1_pi_phase() {
    let g = U1::new(PI);
    let matrix = g.matrix;

    let zero = QLangComplex::zero();
    let one = QLangComplex::one();

    assert_eq!(matrix[[0, 0]], one);
    assert_eq!(matrix[[0, 1]], zero);
    assert_eq!(matrix[[1, 0]], zero);
    assert_eq!(matrix[[1, 1]], QLangComplex::from_polar(1.0, PI));
}
