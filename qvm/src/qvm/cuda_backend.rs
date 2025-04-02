#![cfg(feature = "cuda")]
use cust::{context::Context, memory::*, prelude::*, stream::Stream};


use rand::Rng;

use crate::qvm::cuda::{types::CudaComplex, executor::{launch_cuda_gate_kernel, KernelArg}};
use crate::qvm::backend::QuantumBackend;
use crate::qvm::util::{infer_theta_from_matrix, get_cuda_gate_kernel};
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;


pub struct CudaBackend {
    context: Context,
    _device: Device,
    _stream: Stream,
    state: DeviceBuffer<CudaComplex>,
    num_qubits: usize,
}

impl CudaBackend {
    pub fn new(num_qubits: usize) -> Self {
        // Inicializa o CUDA
        cust::init(cust::CudaFlags::empty()).unwrap();

        let device = Device::get_device(0).unwrap();
        let context = Context::new(device).unwrap();
        let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

        let state_len = 1 << num_qubits;
        let mut host_state = Self::default_host_state(num_qubits);
        host_state[0] = CudaComplex::new(1.0, 0.0); // |0⟩

        let state = DeviceBuffer::from_slice(&host_state).unwrap();

        Self {
            context,
            _device: device,
            _stream: stream,
            state,
            num_qubits,
        }
    }

    fn default_host_state(num_qubits: usize) -> Vec<CudaComplex> {
        let state_len = 1 << num_qubits;
        let mut tmp = vec![CudaComplex::default(); state_len];
        tmp[0] = CudaComplex { re: 1.0, im: 0.0 };
        tmp
    }
}

impl QuantumBackend for CudaBackend {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn state_vector(&self) -> Vec<CudaComplex> {
        let mut host = vec![CudaComplex::default(); self.state.len()];
        self.state.copy_to(&mut host).unwrap();
        host.into_iter().map(CudaComplex::from).collect()
    }

    fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        use crate::qvm::cuda::executor::{launch_cuda_gate_kernel, KernelArg};

        let name = gate.name();

        if let Some(kernel) = get_cuda_gate_kernel(name) {
            let mut args = vec![
                KernelArg::Ptr(self.state.as_device_ptr()),
                KernelArg::I32(qubit as i32),
                KernelArg::I32(self.num_qubits as i32),
            ];

            if kernel.use_theta {
                // usa a matriz para extrair o theta
                let matrix = gate.matrix();
                let theta = infer_theta_from_matrix(&matrix); // você pode definir essa lógica
                args.push(KernelArg::F64(theta));
            }

            launch_cuda_gate_kernel(
                kernel.kernel_name,
                kernel.ptx_file,
                &args,
                &self._stream,
                &self.context,
            );

            return;
        }

        // Fallback: CPU
        let mut host = vec![CudaComplex::default(); self.state.len()];
        self.state.copy_to(&mut host).unwrap();

        let dim = host.len();
        let gate_matrix = gate.matrix();
        let mut new_state = host.clone();

        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                let j = i ^ (1 << qubit);
                let a = host[i];
                let b = host[j];
                new_state[i] =
                    CudaComplex::from(gate_matrix[(0, 0)]) * a + CudaComplex::from(gate_matrix[(0, 1)]) * b;
                new_state[j] =
                    CudaComplex::from(gate_matrix[(1, 0)]) * a + CudaComplex::from(gate_matrix[(1, 1)]) * b;
            }
        }

        self.state.copy_from(&new_state).unwrap();
    }

    fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize) {
        let mut host = vec![CudaComplex::default(); self.state.len()];
        self.state.copy_to(&mut host).unwrap();

        let n = self.num_qubits;
        let dim = 1 << n;
        let gate_matrix = gate.matrix();

        let mut new_state = vec![CudaComplex::default(); dim];

        for i in 0..dim {
            let b1 = (i >> q1) & 1;
            let b2 = (i >> q2) & 1;
            let input_index = (b1 << 1) | b2;

            for output_index in 0..4 {
                let o1 = (output_index >> 1) & 1;
                let o2 = output_index & 1;

                let mut j = i;
                j &= !(1 << q1);
                j &= !(1 << q2);
                j |= o1 << q1;
                j |= o2 << q2;

                let amp = gate_matrix[(output_index, input_index)];
                new_state[j] = new_state[j] + CudaComplex::from(amp) * host[i];
            }
        }

        self.state.copy_from(&new_state).unwrap();
    }

    fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        let name = gate.name();
        let n = self.num_qubits;

        assert!(q0 < n && q1 < n && q2 < n && q0 != q1 && q0 != q2 && q1 != q2,
            "Gate de 3 qubits requer índices distintos e válidos (0..{})", n - 1);

        if let Some(kernel) = get_cuda_gate_kernel(name) {
            let mut args = vec![
                KernelArg::Ptr(self.state.as_device_ptr()),
                KernelArg::I32(q0 as i32),
                KernelArg::I32(q1 as i32),
                KernelArg::I32(q2 as i32),
                KernelArg::I32(n as i32),
            ];

            if kernel.use_theta {
                let matrix = gate.matrix();
                let theta = infer_theta_from_matrix(&matrix);
                args.push(KernelArg::F64(theta));
            }

            launch_cuda_gate_kernel(
                kernel.kernel_name,
                kernel.ptx_file,
                &args,
                &self._stream,
                &self.context,
            );

            return;
        }

        // Fallback: CPU-style simulação com host (como nos 2Q)
        let mut host = vec![CudaComplex::default(); self.state.len()];
        self.state.copy_to(&mut host).unwrap();

        let dim = 1 << n;
        let gate_matrix = gate.matrix();
        let mut new_state = host.clone();

        for i in 0..dim {
            let b0 = (i >> q0) & 1;
            let b1 = (i >> q1) & 1;
            let b2 = (i >> q2) & 1;
            let input_index = (b0 << 2) | (b1 << 1) | b2;

            for output_index in 0..8 {
                let o0 = (output_index >> 2) & 1;
                let o1 = (output_index >> 1) & 1;
                let o2 = output_index & 1;

                let mut j = i;
                j &= !(1 << q0);
                j &= !(1 << q1);
                j &= !(1 << q2);
                j |= o0 << q0;
                j |= o1 << q1;
                j |= o2 << q2;

                let amp = gate_matrix[(output_index, input_index)];
                new_state[j] = new_state[j] + CudaComplex::from(amp) * host[i];
            }
        }

        self.state.copy_from(&new_state).unwrap();
    }


    fn measure_all(&mut self) -> Vec<u8> {
        let mut results = vec![0; self.num_qubits];
        for qubit in 0..self.num_qubits {
            results[qubit] = self.measure(qubit);
        }
        results
    }

    fn measure(&mut self, qubit: usize) -> u8 {
        let dim = self.state.len();
        let mut host = vec![CudaComplex::default(); dim];
        self.state.copy_to(&mut host).unwrap();

        let mut prob_0 = 0.0;
        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                prob_0 += host[i].norm_sqr();
            }
        }

        let mut rng = rand::thread_rng();
        let rand_value: f64 = rng.gen();

        let measured_value = if rand_value < prob_0 { 0 } else { 1 };

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit != measured_value {
                host[i] = CudaComplex::new(0.0, 0.0);
            }
        }

        let norm_factor = (host.iter().map(|x| x.norm_sqr()).sum::<f64>()).sqrt();
        let norm_complex = CudaComplex::new(norm_factor, 0.0);
        for h in &mut host {
            *h /= norm_complex;
        }

        self.state.copy_from(&host).unwrap();
        measured_value as u8
    }

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
            *self = CudaBackend::new(num_qubits);
        } else {
            let mut host_state = Self::default_host_state(num_qubits);
            host_state[0] = CudaComplex::new(1.0, 0.0);
            self.state.copy_from(&host_state).unwrap();
        }
    }

}
pub type Backend = CudaBackend; 