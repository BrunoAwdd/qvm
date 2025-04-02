use qlang::gates::ry::RY;
use qlang::qvm::cuda::types::CudaComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_ry_pi() {
    let ry = RY::new(PI);
    let matrix = ry.matrix;
    let expected = array![
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(-1.0, 0.0)],
        [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
    ];
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
}
