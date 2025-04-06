use qlang::gates::one_q::pauli_y::PauliY;
use qlang::qvm::cuda::types::CudaComplex;
use ndarray::array;

#[test]
fn test_pauli_y_matrix() {
    let y = PauliY::new();

    let expected = array![
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(0.0, -1.0)],
        [CudaComplex::new(0.0, 1.0), CudaComplex::new(0.0, 0.0)]
    ];

    assert_eq!(y.matrix, expected);
}
