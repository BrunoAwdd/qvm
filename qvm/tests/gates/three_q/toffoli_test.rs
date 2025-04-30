use qlang::gates::three_q::toffoli::Toffoli;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_toffoli_matrix_dimensions() {
    let g = Toffoli::new();
    let matrix = g.matrix;

    let one = QLangComplex::one();

    assert_eq!(matrix.nrows(), 8);
    assert_eq!(matrix.ncols(), 8);

    // |110⟩ → |111⟩  → pos 6 → 7
    assert_eq!(matrix[[7, 6]], one);
    // |111⟩ → |110⟩  → pos 7 → 6
    assert_eq!(matrix[[6, 7]], one);
}
