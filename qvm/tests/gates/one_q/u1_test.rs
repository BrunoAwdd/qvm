use qlang::gates::one_q::u1::U1;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;

#[test]
fn test_u1_pi_phase() {
    let g = U1::new(PI);
    let matrix = g.matrix;

    assert_eq!(matrix[[0, 0]], QLangComplex::new(1.0, 0.0));
    assert_eq!(matrix[[0, 1]], QLangComplex::new(0.0, 0.0));
    assert_eq!(matrix[[1, 0]], QLangComplex::new(0.0, 0.0));
    assert_eq!(matrix[[1, 1]], QLangComplex::from_polar(1.0, PI));
}
