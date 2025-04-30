use ndarray::{Array2, array};
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct ISwap {
    matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for ISwap {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "iswap"
    }
}

impl ISwap {
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let i = QLangComplex::i();

        let matrix = array![
            [one,  zero, zero, zero],
            [zero, zero,   i,  zero],
            [zero,   i,  zero, zero],
            [zero, zero, zero, one],
        ];

        Self { matrix }
    }
}
