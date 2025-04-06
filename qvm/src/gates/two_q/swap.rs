use ndarray::array;
use ndarray::Array2;

use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Swap {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Swap {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "SWAP"
    }
}

impl Swap {
    pub fn new() -> Self {
        let zero = QLangComplex::new(0.0, 0.0);
        let one = QLangComplex::new(1.0, 0.0);

        let matrix = array![
            [one, zero, zero, zero],
            [zero, zero, one, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
        ];

        Self { matrix }
    }
}
