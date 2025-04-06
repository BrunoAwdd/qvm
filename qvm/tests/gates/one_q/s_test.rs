use qlang::gates::one_q::s::S;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_s_gate_matrix() {
    let s = S::new();
    let matrix = s.matrix;
    assert_eq!(matrix[[0, 0]], QLangComplex::new(1.0, 0.0));
    assert_eq!(matrix[[1, 1]], QLangComplex::new(0.0, 1.0));
}
