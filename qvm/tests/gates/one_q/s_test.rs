use qlang::gates::one_q::s::S;
use qlang::qvm::cuda::types::CudaComplex;

#[test]
fn test_s_gate_matrix() {
    let s = S::new();
    let matrix = s.matrix;
    assert_eq!(matrix[[0, 0]], CudaComplex::new(1.0, 0.0));
    assert_eq!(matrix[[1, 1]], CudaComplex::new(0.0, 1.0));
}
