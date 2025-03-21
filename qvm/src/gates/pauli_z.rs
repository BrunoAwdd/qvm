use ndarray::array;
use ndarray::Array2;
use num_complex::Complex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

pub struct PauliZ {
    pub matrix: Array2<Complex<f64>>,
}

impl QuantumGateAbstract for PauliZ {
    fn matrix(&self) -> Array2<Complex<f64>> {
        self.matrix.clone()
    }
}

impl PauliZ {
    pub fn new() -> Self {
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<Complex<f64>>, ndarray::Dim<[usize; 2]>> = array![
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)]
        ];

        Self { matrix }
    }
}
