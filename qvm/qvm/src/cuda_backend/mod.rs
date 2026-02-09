// src/qvm/cuda_backend/mod.rs
//#![cfg(feature = "cuda")]

pub mod apply;
pub mod fallback;
pub mod init;
pub mod measure;

use crate::{
    gates::quantum_gate_abstract::QuantumGateAbstract, qvm::backend::QuantumBackend,
    types::qlang_complex::QLangComplex,
};
use cust::{context::Context, device::Device, memory::*, stream::Stream};

pub struct CudaBackend {
    // Context must be dropped last
    pub context: Context,
    pub _device: Device,
    pub _stream: Stream,
    pub state: DeviceBuffer<QLangComplex>,
    pub num_qubits: usize,
}

impl QuantumBackend for CudaBackend {
    fn num_qubits(&self) -> usize { self.num_qubits }

    fn state_vector(&self) -> Vec<QLangComplex> {
        let mut host = vec![QLangComplex::default(); self.state.len()];
        self.state
            .copy_to(&mut host)
            .expect("Failed to copy CUDA device buffer to host");
        host
    }

    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        self.apply_gate(gate, qubit);
    }

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        self.apply_gate_2q(gate, q0, q1);
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        self.apply_gate_3q(gate, q0, q1, q2);
    }

    fn measure(&mut self, qubit: usize) -> u8 { self.measure(qubit) }

    fn measure_many(&mut self, qubits: &Vec<usize>) -> Vec<u8> { self.measure_many(qubits) }

    fn measure_all(&mut self) -> Vec<u8> { self.measure_all() }

    fn display(&self) {
        let state = self.state_vector();
        println!("Quantum State with {} qubits:", self.num_qubits);
        for (i, amp) in state.iter().enumerate() {
            if amp.norm_sqr() > 1e-10 {
                println!(
                    "|{:0width$b}⟩: {:.4} + {:.4}i",
                    i,
                    amp.re,
                    amp.im,
                    width = self.num_qubits
                );
            }
        }
    }

    fn reset(&mut self, num_qubits: usize) {
        if num_qubits != self.num_qubits {
            let mut new_backend = CudaBackend::new(num_qubits);
            std::mem::swap(self, &mut new_backend); // swap seguro
                                                    // `new_backend` é dropado ao final do escopo
        } else {
            let host_state = Self::default_host_state(num_qubits);
            self.state
                .copy_from(&host_state)
                .map_err(|e| println!("Error reseting state: {:?}", e))
                .expect("Failed to copy host to CUDA device buffer");
        }
    }

    fn box_clone(&self) -> Box<dyn QuantumBackend> {
        let cuda_backend = CudaBackend::new(self.num_qubits);
        Box::new(cuda_backend)
    }

    fn name(&self) -> &'static str { "CUDA" }
}
impl Drop for CudaBackend {
    fn drop(&mut self) {
        println!("🧹 Drop: sincronizando stream antes de liberar");
        let _ = self._stream.synchronize();
        // Buffers, stream e contexto serão dropados automaticamente.
        println!("🧹 Drop: backend CUDA liberado (contexto, buffers e stream dropados)");
    }
}
impl Clone for CudaBackend {
    fn clone(&self) -> Self {
        let mut host_state = vec![QLangComplex::default(); self.state.len()];
        self.state
            .copy_to(&mut host_state)
            .expect("Failed to copy CUDA device buffer to host");

        let mut new_backend = CudaBackend::new(self.num_qubits);
        new_backend
            .state
            .copy_from(&host_state)
            .expect("Failed to copy host to CUDA device buffer");
        new_backend
    }
}

pub type Backend = CudaBackend;
