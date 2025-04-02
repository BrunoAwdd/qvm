use ndarray::Array2;
use crate::qvm::cuda::types::CudaComplex;

pub trait QuantumGateAbstract {
    fn matrix(&self) -> Array2<CudaComplex>;
    fn name(&self) -> &'static str;
}
