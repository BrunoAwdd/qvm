
use ndarray::Array2;
use ndarray::array;
use num_complex::Complex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;


pub struct Hadamard {
    pub matrix: Array2<Complex<f64>>,
}

impl QuantumGateAbstract for Hadamard {
    fn matrix(&self) -> Array2<Complex<f64>> {
        self.matrix.clone()
    }
}

impl Hadamard {
    pub fn new() -> Self {
        let factor: f64 = 1.0 / (2.0_f64).sqrt();
        let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<Complex<f64>>, ndarray::Dim<[usize; 2]>> = array![
            [Complex::new(factor, 0.0), Complex::new(factor, 0.0)],
            [Complex::new(factor, 0.0), Complex::new(-factor, 0.0)]
        ];

        Self { matrix }
    }
}

