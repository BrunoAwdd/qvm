use ndarray::array;
use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct CNOT {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for CNOT {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "CNOT"
    }
}

impl CNOT {
    pub fn new() -> Self {
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<QLangComplex>, ndarray::Dim<[usize; 2]>> = array![
            [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0)],
            [QLangComplex::new(0.0, 0.0), QLangComplex::new(0.0, 0.0), QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)]
        ];

        Self { matrix }
    }
}
