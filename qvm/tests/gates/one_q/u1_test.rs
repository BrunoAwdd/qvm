use qlang::gates::one_q::u1::U1;
use qlang::qvm::cuda::types::CudaComplex;
use std::f64::consts::PI;

#[test]
fn test_u1_pi_phase() {
    let g = U1::new(PI);
    let matrix = g.matrix;

    assert_eq!(matrix[[0, 0]], CudaComplex::new(1.0, 0.0));
    assert_eq!(matrix[[0, 1]], CudaComplex::new(0.0, 0.0));
    assert_eq!(matrix[[1, 0]], CudaComplex::new(0.0, 0.0));
    assert_eq!(matrix[[1, 1]], CudaComplex::from_polar(1.0, PI));
}
