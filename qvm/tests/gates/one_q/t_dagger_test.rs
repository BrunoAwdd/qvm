use qlang::gates::one_q::t_dagger::TDagger;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::FRAC_PI_4; // π/4
use ndarray::array;

#[test]
fn test_t_dagger_matrix() {
    let tdg = TDagger::new();
    let matrix = tdg.matrix;
    let expected = array![
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::from_polar(1.0, -FRAC_PI_4)],
    ];
    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
