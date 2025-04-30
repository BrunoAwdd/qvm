use ndarray::Array2;
use ndarray_linalg::krylov::Q;
use num_traits::Zero;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct ControlledU {
    matrix: Array2<QLangComplex>,
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

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::from_elem((4, 4), zero);
        matrix[[0, 0]] = one;
        matrix[[1, 1]] = one;

        for i in 0..2 {
            for j in 0..2 {
                matrix[[i + 2, j + 2]] = u[[i, j]];
            }
        }

        Self { matrix }
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
