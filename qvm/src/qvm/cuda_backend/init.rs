// src/qvm/cuda_backend/init.rs
#![cfg(feature = "cuda")]
use super::CudaBackend;
use crate::types::qlang_complex::QLangComplex;
use cust::{
    context::Context, 
    device::Device, 
    memory::*, 
    stream::{Stream, StreamFlags}
};

impl CudaBackend {
    pub fn new(num_qubits: usize) -> Self {
        println!("CudaBackend: network created with {} qubits", num_qubits);
        cust::init(cust::CudaFlags::empty()).expect("CUDA init failed");
        let device = Device::get_device(0).unwrap();
        let context = Context::new(device).unwrap();
        let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

        let host_state = Self::default_host_state(num_qubits);
        let state = DeviceBuffer::from_slice(&host_state).unwrap();

        Self {
            context,
            _device: device,
            _stream: stream,
            state,
            num_qubits,
        }
    }

    pub fn default_host_state(num_qubits: usize) -> Vec<QLangComplex> {
        let mut state = vec![QLangComplex::default(); 1 << num_qubits];
        state[0] = one;
        state
    }
}
