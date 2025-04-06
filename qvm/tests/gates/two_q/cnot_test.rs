use qlang::gates::two_q::cnot::CNOT;
use qlang::types::qlang_complex::QLangComplex;
use ndarray::array;

#[test]
fn test_cnot_matrix() {
    let cnot = CNOT::new();

    let expected = array![
        [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0)],
        [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)]
    ];

    assert_eq!(cnot.matrix, expected);
}
