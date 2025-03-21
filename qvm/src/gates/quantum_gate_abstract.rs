use ndarray::Array2;
use num_complex::Complex;

pub trait QuantumGateAbstract {
    fn matrix(&self) -> Array2<Complex<f64>>;
}
