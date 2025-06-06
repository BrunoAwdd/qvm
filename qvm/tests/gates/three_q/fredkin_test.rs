use qlang::gates::three_q::fredkin::Fredkin;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_fredkin_matrix_dimensions() {
    let g = Fredkin::new();
    let matrix = g.matrix;

    let one = QLangComplex::one();

    assert_eq!(matrix.nrows(), 8);
    assert_eq!(matrix.ncols(), 8);

    // |110⟩ ↔ |101⟩ (control = 1, troca qubits 1 e 2)
    assert_eq!(matrix[[5, 6]], one);
    assert_eq!(matrix[[6, 5]], one);
}
