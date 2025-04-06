use qlang::gates::one_q::t::T;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;

#[test]
fn test_t_gate_matrix() {
    let t = T::new();
    let matrix = t.matrix;
    let angle = PI / 4.0;
    let expected = QLangComplex::new(angle.cos(), angle.sin());
    assert_eq!(matrix[[1, 1]], expected);
}
