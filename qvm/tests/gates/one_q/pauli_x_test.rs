use qlang::gates::one_q::pauli_x::PauliX; // Importação correta do módulo
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_pauli_x_matrix() {
    let x = PauliX::new();

    let zero = QLangComplex::zero();
    let one = QLangComplex::one();

    let expected = array![
        [zero, one],
        [one, zero]
    ];

    assert_eq!(x.matrix, expected);
}
