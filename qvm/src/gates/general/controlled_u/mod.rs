use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct ControlledU {
    matrix: Array2<QLangComplex>,
    _name: String, // mantém caso você queira usar no futuro
}

impl QuantumGateAbstract for ControlledU {
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    fn name(&self) -> &'static str {
        "cu"
    }
}

impl ControlledU {
    pub fn new(u: Array2<QLangComplex>, name: Option<String>) -> Self {
        assert_eq!(u.dim(), (2, 2), "ControlledU expects a 2x2 unitary matrix");

        let zero = QLangComplex::new(0.0, 0.0);
        let one = QLangComplex::new(1.0, 0.0);

        let mut cu = Array2::from_elem((4, 4), zero);
        cu[[0, 0]] = one;
        cu[[1, 1]] = one;

        for i in 0..2 {
            for j in 0..2 {
                cu[[i + 2, j + 2]] = u[[i, j]];
            }
        }

        Self {
            matrix: cu,
            _name: name.unwrap_or_else(|| "cu".to_string()),
        }
    }

    pub fn new_real(u00: f64, u01: f64, u10: f64, u11: f64) -> Self {
        use ndarray::array;
        let u = array![
            [QLangComplex::new(u00, 0.0), QLangComplex::new(u01, 0.0)],
            [QLangComplex::new(u10, 0.0), QLangComplex::new(u11, 0.0)],
        ];
        Self::new(u, None)
    }
}
