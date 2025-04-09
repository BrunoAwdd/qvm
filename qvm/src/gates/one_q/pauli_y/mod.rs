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
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<QLangComplex>, ndarray::Dim<[usize; 2]>> = array![
            [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, -1.0)],
            [QLangComplex::new(0.0, 1.0), QLangComplex::new(0.0, 0.0)]
        ];

        Self { matrix }
    }
}
