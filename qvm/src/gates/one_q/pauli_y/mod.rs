use ndarray::array;
use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct PauliY {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for PauliY {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "pauliY"
    }
}

impl PauliY {
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let i = QLangComplex::i();
        let neg_i = QLangComplex::neg_i();

        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<QLangComplex>, ndarray::Dim<[usize; 2]>> = array![
            [zero, neg_i],
            [i, zero]
        ];

        Self { matrix }
    }
}
