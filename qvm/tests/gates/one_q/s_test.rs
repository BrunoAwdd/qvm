use qlang::gates::one_q::s::S;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_s_gate_matrix() {
    let s = S::new();
    let matrix = s.matrix;

    let one = QLangComplex::one();
    let i = QLangComplex::i();

    assert_eq!(matrix[[0, 0]], one);
    assert_eq!(matrix[[1, 1]], i);
}
