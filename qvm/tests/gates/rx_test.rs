use qlang::gates::rx::RX;
use qlang::qvm::cuda::types::CudaComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_rx_pi() {
    let rx = RX::new(PI);
    let matrix = rx.matrix;
    let expected = array![
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, -1.0)],
        [CudaComplex::new(0.0, -1.0), CudaComplex::new(0.0, 0.0)],
    ];
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
}
