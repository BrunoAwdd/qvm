// src/qvm/cuda_backend/apply.rs
#![cfg(feature = "cuda")]
use super::CudaBackend;
use crate::{
    gates::quantum_gate_abstract::QuantumGateAbstract,
    qvm::{ 
        cuda::executor::{launch_cuda_gate_kernel, KernelArg},
        util::{get_cuda_gate_kernel, infer_theta_from_matrix},
    },
};

impl CudaBackend {
    pub fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        let name = gate.name();
        if let Some(kernel) = get_cuda_gate_kernel(name) {
            let mut args = vec![
                KernelArg::Ptr(self.state.as_device_ptr()),
                KernelArg::I32(qubit as i32),
                KernelArg::I32(self.num_qubits as i32),
            ];

            if kernel.use_theta {
                if name == "U3" {
                    let (theta, phi, lambda) = gate
                        .as_u3_params()
                        .expect("U3 gate não implementa `as_u3_params`.");
                    args.extend([theta, phi, lambda].map(KernelArg::F64));
                } else {
                    let theta = infer_theta_from_matrix(&gate.matrix());
                    args.push(KernelArg::F64(theta));
                }
            }

            launch_cuda_gate_kernel(
                kernel.kernel_name,
                kernel.ptx_file,
                &args,
                &self._stream,
                &self.context,
            );
        } else {
            self.execute_fallback(&gate.matrix(), &[qubit]);
        }
    }

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q1: usize, q2: usize) {
        self.execute_fallback(&gate.matrix(), &[q1, q2]);
    }

    pub fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        let n = self.num_qubits;
        assert!(q0 < n && q1 < n && q2 < n && q0 != q1 && q0 != q2 && q1 != q2,
            "Gate de 3 qubits requer índices distintos e válidos (0..{})", n - 1);

        if let Some(kernel) = get_cuda_gate_kernel(gate.name()) {
            let mut args = vec![
                KernelArg::Ptr(self.state.as_device_ptr()),
                KernelArg::I32(q0 as i32),
                KernelArg::I32(q1 as i32),
                KernelArg::I32(q2 as i32),
                KernelArg::I32(n as i32),
            ];

            if kernel.use_theta {
                let theta = infer_theta_from_matrix(&gate.matrix());
                args.push(KernelArg::F64(theta));
            }

            launch_cuda_gate_kernel(
                kernel.kernel_name,
                kernel.ptx_file,
                &args,
                &self._stream,
                &self.context,
            );
        } else {
            self.execute_fallback(&gate.matrix(), &[q0, q1, q2]);
        }
    }
}
