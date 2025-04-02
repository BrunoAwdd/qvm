use qlang::gates::cnot::CNOT;
use qlang::qvm::cuda::types::CudaComplex;
use ndarray::array;

#[test]
fn test_cnot_matrix() {
    let cnot = CNOT::new();

    let expected = array![
        [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0)],
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0)],
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0)],
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)]
    ];

    assert_eq!(cnot.matrix, expected);
}
