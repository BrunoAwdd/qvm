use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;
use ndarray::{Array2, array};

pub struct SDagger {
    pub matrix: Array2<QLangComplex>,
}

impl SDagger {
    pub fn new() -> Self {

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_i = QLangComplex::neg_i();

        let matrix = array![
            [one, zero],
            [zero, neg_i]
        ];
        Self { matrix }
    }
}

impl QuantumGateAbstract for SDagger {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "SDagger"
    }
}
