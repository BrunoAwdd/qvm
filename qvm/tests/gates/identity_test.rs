use qlang::gates::identity::Identity;
use qlang::qvm::cuda::types::CudaComplex;
use ndarray::array;

#[test]
fn test_identity_gate_matrix() {
    let id = Identity::new();
    let matrix = id.matrix;

    let expected = array![
        [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)],
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0)],
    ];

    assert_eq!(matrix[[0, 0]], expected[[0, 0]]);
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
    assert_eq!(matrix[[1, 1]], expected[[1, 1]]);
}
