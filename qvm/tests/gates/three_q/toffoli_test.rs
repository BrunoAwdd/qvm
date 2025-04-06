use qlang::gates::three_q::toffoli::Toffoli;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_toffoli_matrix_dimensions() {
    let g = Toffoli::new();
    let matrix = g.matrix;

    assert_eq!(matrix.nrows(), 8);
    assert_eq!(matrix.ncols(), 8);

    // |110⟩ → |111⟩  → pos 6 → 7
    assert_eq!(matrix[[7, 6]], QLangComplex::new(1.0, 0.0));
    // |111⟩ → |110⟩  → pos 7 → 6
    assert_eq!(matrix[[6, 7]], QLangComplex::new(1.0, 0.0));
}
