use qlang::gates::one_q::pauli_y::PauliY;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_pauli_y_matrix() {
    let y = PauliY::new();
    
    let zero = QLangComplex::zero();
    let i = QLangComplex::i();
    let neg_i = QLangComplex::neg_i();

    let expected = array![
        [zero, neg_i],
        [i, zero]
    ];

    assert_eq!(y.matrix, expected);
}
