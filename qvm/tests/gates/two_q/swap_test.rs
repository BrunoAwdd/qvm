use qlang::gates::two_q::swap::Swap;
use qlang::types::qlang_complex::QLangComplex;

#[test]
fn test_swap_matrix() {
    let swap = Swap::new();
    let m = swap.matrix;

    assert_eq!(m[[0, 0]], QLangComplex::new(1.0, 0.0));
    assert_eq!(m[[1, 2]], QLangComplex::new(1.0, 0.0));
    assert_eq!(m[[2, 1]], QLangComplex::new(1.0, 0.0));
    assert_eq!(m[[3, 3]], QLangComplex::new(1.0, 0.0));
}
