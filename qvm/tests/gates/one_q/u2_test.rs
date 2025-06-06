use qlang::gates::rotation_q::u2::U2;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;

#[test]
fn test_u2_pi_params() {
    let g = U2::new(0.0, PI);
    let matrix = g.matrix;

    let sqrt_2_inv = 1.0 / f64::sqrt(2.0);
    let e_i_pi = QLangComplex::from_polar(1.0, PI); // -1 + 0i
    let e_i_0 = QLangComplex::from_polar(1.0, 0.0); // 1 + 0i
    let e_i_sum = QLangComplex::from_polar(1.0, PI); // phi + lambda = π

    assert_eq!(matrix[[0, 0]], QLangComplex::new(sqrt_2_inv, 0.0));
    assert_eq!(matrix[[0, 1]], -e_i_pi * sqrt_2_inv);
    assert_eq!(matrix[[1, 0]], e_i_0 * sqrt_2_inv);
    assert_eq!(matrix[[1, 1]], e_i_sum * sqrt_2_inv);
}
