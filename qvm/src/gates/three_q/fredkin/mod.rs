use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct Fredkin {
    pub matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for Fredkin {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }
    fn name(&self) -> &'static str {
        "Fredkin"
    }
}

impl Fredkin {
    pub fn new() -> Self {
        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::<QLangComplex>::eye(8);
        matrix[[5, 5]] = zero;
        matrix[[6, 6]] = zero;
        matrix[[5, 6]] = one;
        matrix[[6, 5]] = one;
        Self { matrix }
    }
}
