use qlang::gates::rotation_q::rx::RX;
use qlang::types::qlang_complex::QLangComplex;
use std::f64::consts::PI;
use ndarray::array;

#[test]
fn test_rx_pi() {
    let rx = RX::new(PI);
    let matrix = rx.matrix;

    let zero = QLangComplex::zero();
    let neg_i = QLangComplex::neg_i();

    let expected = array![
        [zero, neg_i],
        [neg_i, zero],
    ];
    assert_eq!(matrix[[0, 1]], expected[[0, 1]]);
    assert_eq!(matrix[[1, 0]], expected[[1, 0]]);
}
