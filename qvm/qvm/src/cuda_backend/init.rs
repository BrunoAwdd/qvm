// src/qvm/cuda_backend/init.rs
//#![cfg(feature = "cuda")]
use super::CudaBackend;
use crate::types::qlang_complex::QLangComplex;
use cust::{
    context::Context,
    device::Device,
    memory::*,
    stream::{Stream, StreamFlags},
};

impl CudaBackend {
    pub fn new(num_qubits: usize) -> Self {
        println!(
            "CudaBackend: network created with {} qubits (Begin)\n",
            num_qubits
        );
        cust::init(cust::CudaFlags::empty()).expect("CUDA init failed msg");

        let device_count = Device::num_devices().expect("Failed to get CUDA device count");
        if device_count == 0 {
            panic!("No CUDA devices found — aborting.");
        }

        let device = Device::get_device(0).expect("No CUDA device 0 available");
        let context = Context::new(device).expect("Failed to create CUDA context");
        let stream = Stream::new(StreamFlags::DEFAULT, None).expect("Failed to create CUDA stream");

        let host_state = Self::default_host_state(num_qubits);
        let state =
            DeviceBuffer::from_slice(&host_state).expect("Failed to create CUDA device buffer");

        println!(
            "[CudaBackend] device buffer alocado: {} elementos ({} bytes)",
            host_state.len(),
            std::mem::size_of::<QLangComplex>() * host_state.len()
        );

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
        let one = QLangComplex::one();
        state[0] = one;
        state
    }
}
