// src/state/tensor_network.rs
use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;


pub fn contract(a: Array2<QLangComplex>, b: Array2<QLangComplex>) -> Array2<QLangComplex> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        panic!(
            "❌ Erro de contração: state = {:?}, tensor = {:?} — dim compartilhada não bate: {} != {}",
            a.dim(),
            b.dim(),
            k1,
            k2
        );
    }

    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            for k in 0..k1 {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    result
}