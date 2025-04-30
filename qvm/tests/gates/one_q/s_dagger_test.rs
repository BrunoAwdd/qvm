use qlang::gates::one_q::s_dagger::SDagger;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::FRAC_PI_2; // π/2
use ndarray::array;

#[test]
fn test_s_dagger_matrix() {
    let sdg = SDagger::new();
    let matrix = sdg.matrix;

    let zero = QLangComplex::zero();
    let one = QLangComplex::one();

    let expected = array![
        [one, zero],
        [zero, QLangComplex::from_polar(1.0, -FRAC_PI_2)],
    ];
    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
