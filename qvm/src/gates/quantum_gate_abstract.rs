use ndarray::Array2;
use crate::qvm::cuda::types::CudaComplex;
use std::any::Any;

pub trait QuantumGateAbstract: Any {
    fn matrix(&self) -> Array2<CudaComplex>;
    fn name(&self) -> &'static str;

    fn as_u3_params(&self) -> Option<(f64, f64, f64)> {
        None
    }
}
