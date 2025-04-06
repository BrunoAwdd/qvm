use qlang::gates::one_q::pauli_x::PauliX; // Importação correta do módulo
use qlang::qvm::cuda::types::CudaComplex;
use ndarray::array;

#[test]
fn test_pauli_x_matrix() {
    let x = PauliX::new();

    let expected = array![
        [CudaComplex::new(0.0, 0.0), CudaComplex::new(1.0, 0.0)],
        [CudaComplex::new(1.0, 0.0), CudaComplex::new(0.0, 0.0)]
    ];

    assert_eq!(x.matrix, expected);
}
