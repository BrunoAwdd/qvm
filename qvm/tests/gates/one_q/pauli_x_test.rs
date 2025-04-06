use qlang::gates::one_q::pauli_x::PauliX; // Importação correta do módulo
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_pauli_x_matrix() {
    let x = PauliX::new();

    let expected = array![
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0)],
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)]
    ];

    assert_eq!(x.matrix, expected);
}
