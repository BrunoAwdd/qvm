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
        let mut mat = Array2::<QLangComplex>::eye(8);
        mat[[5, 5]] = QLangComplex::new(0.0, 0.0);
        mat[[6, 6]] = QLangComplex::new(0.0, 0.0);
        mat[[5, 6]] = QLangComplex::new(1.0, 0.0);
        mat[[6, 5]] = QLangComplex::new(1.0, 0.0);
        Self { matrix: mat }
    }
}
