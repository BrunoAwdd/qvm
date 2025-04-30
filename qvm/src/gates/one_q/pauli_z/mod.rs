use ndarray::array;
use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct PauliZ {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for PauliZ {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "PauliZ"
    }
}

impl PauliZ {
    pub fn new() -> Self {

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();
        let neg_one = QLangComplex::neg_one();

        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<QLangComplex>, ndarray::Dim<[usize; 2]>> = array![
            [one, zero],
            [zero, neg_one]
        ];

        Self { matrix }
    }
}
