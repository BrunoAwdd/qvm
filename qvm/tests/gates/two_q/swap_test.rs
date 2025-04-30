use qlang::gates::two_q::swap::Swap;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_swap_matrix() {
    let swap = Swap::new();
    let m = swap.matrix;

    let one = QLangComplex::one();

    assert_eq!(m[[0, 0]], one);
    assert_eq!(m[[1, 2]], one);
    assert_eq!(m[[2, 1]], one);
    assert_eq!(m[[3, 3]], one);
}
