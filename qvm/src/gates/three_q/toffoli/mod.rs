use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Toffoli {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Toffoli {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Toffoli"
    }
}

impl Toffoli {
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::<QLangComplex>::eye(8);
        matrix[[6, 6]] = zero;
        matrix[[7, 7]] = zero;
        matrix[[6, 7]] = one;
        matrix[[7, 6]] = one;
        Self { matrix }
    }
}
