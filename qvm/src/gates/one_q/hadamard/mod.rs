
use ndarray::Array2;
use ndarray::array;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;


pub struct Hadamard {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Hadamard {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Hadamard"
    }
}

impl Hadamard {
    pub fn new() -> Self {
        let factor: f64 = 1.0 / (2.0_f64).sqrt();
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<QLangComplex>, ndarray::Dim<[usize; 2]>> = array![
            [QLangComplex::new(factor, 0.0), QLangComplex::new(factor, 0.0)],
            [QLangComplex::new(factor, 0.0), QLangComplex::new(-factor, 0.0)]
        ];

        Self { matrix }
    }
}

