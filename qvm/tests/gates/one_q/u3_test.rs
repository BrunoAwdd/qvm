use qlang::gates::rotation_q::u3::U3;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;

#[test]
fn test_u3_pi_params() {
    let g = U3::new(PI, 0.0, 0.0);
    let matrix = g.matrix;

    let cos = (PI / 2.0).cos();
    let sin = (PI / 2.0).sin();

    assert_eq!(matrix[[0, 0]], QLangComplex::new(cos, 0.0));
    assert_eq!(matrix[[1, 0]], QLangComplex::new(sin, 0.0));
    assert_eq!(matrix[[0, 1]], QLangComplex::new(-sin, 0.0));
    assert_eq!(matrix[[1, 1]], QLangComplex::new(cos, 0.0));
}
