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
        let mut mat = Array2::<QLangComplex>::eye(8);
        mat[[6, 6]] = QLangComplex::new(0.0, 0.0);
        mat[[7, 7]] = QLangComplex::new(0.0, 0.0);
        mat[[6, 7]] = QLangComplex::new(1.0, 0.0);
        mat[[7, 6]] = QLangComplex::new(1.0, 0.0);
        Self { matrix: mat }
    }
}
