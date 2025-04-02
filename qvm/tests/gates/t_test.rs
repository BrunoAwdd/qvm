use qlang::gates::t::T;
use qlang::qvm::cuda::types::CudaComplex;
use std::f64::consts::PI;

#[test]
fn test_t_gate_matrix() {
    let t = T::new();
    let matrix = t.matrix;
    let angle = PI / 4.0;
    let expected = CudaComplex::new(angle.cos(), angle.sin());
    assert_eq!(matrix[[1, 1]], expected);
}
