use qlang::gates::two_q::cnot::CNOT;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_cnot_matrix() {
    let cnot = CNOT::new();

    let zero = QLangComplex::zero();
    let one = QLangComplex::one();

    let expected = array![
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
        [zero, zero, one, zero]
    ];

    assert_eq!(cnot.matrix, expected);
}
