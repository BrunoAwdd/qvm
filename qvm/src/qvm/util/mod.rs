
use crate::qvm::cuda::types::CudaComplex;
use ndarray::Array2;

pub struct GateKernel {
    pub kernel_name: &'static str,
    pub ptx_file: &'static str,
    pub use_theta: bool,
}



/// Extrai `θ` de uma matriz de rotação RX, RY ou RZ.
/// Supõe que a matriz é válida e segue o padrão dos gates quânticos.
pub fn infer_theta_from_matrix(matrix: &Array2<CudaComplex>) -> f64 {
    use std::f64::consts::PI;

    let m00 = matrix[(0, 0)];
    let m01 = matrix[(0, 1)];
    let m10 = matrix[(1, 0)];
    let m11 = matrix[(1, 1)];

    // RZ: matriz diagonal e unitária => infere θ a partir da fase relativa
    if m01 == CudaComplex::new(0.0, 0.0) && m10 == CudaComplex::new(0.0, 0.0) {
        // m11 = e^{-iθ/2}, m00 = e^{iθ/2}
        let phase_diff = m11.arg() - m00.arg();
        return -phase_diff;
    }

    // RX e RY têm senos em off-diagonal
    if m00.re.abs() <= 1.0 {
        let theta = 2.0 * m00.re.acos();
        return theta;
    }

    // fallback seguro
    0.0
}

pub fn get_cuda_gate_kernel(gate_name: &str) -> Option<GateKernel> {
    match gate_name {
        "Cnot"     => Some(GateKernel { kernel_name: "cnot_kernel",     ptx_file: "cnot.ptx", use_theta: false }),
        "Fredkin"  => Some(GateKernel { kernel_name: "fredkin_kernel",  ptx_file: "fredkin.ptx", use_theta: false }),
        "Hadamard" => Some(GateKernel { kernel_name: "hadamard_kernel", ptx_file: "hadamard.ptx", use_theta: false }),
        "RX"       => Some(GateKernel { kernel_name: "rx_kernel",       ptx_file: "rx.ptx", use_theta: true  }),
        "RY"       => Some(GateKernel { kernel_name: "ry_kernel",       ptx_file: "ry.ptx", use_theta: true  }),
        "RZ"       => Some(GateKernel { kernel_name: "rz_kernel",       ptx_file: "rz.ptx", use_theta: true  }),
        "PauliX"   => Some(GateKernel { kernel_name: "pauli_x_kernel",  ptx_file: "pauli_x.ptx", use_theta: false }),
        "PauliY"   => Some(GateKernel { kernel_name: "pauli_y_kernel",  ptx_file: "pauli_y.ptx", use_theta: false }),
        "PauliZ"   => Some(GateKernel { kernel_name: "pauli_z_kernel",  ptx_file: "pauli_z.ptx", use_theta: false }),
        "S"        => Some(GateKernel { kernel_name: "s_kernel",        ptx_file: "s.ptx", use_theta: false }),
        "SDagger"  => Some(GateKernel { kernel_name: "sdagger_kernel",  ptx_file: "s_dagger.ptx", use_theta: false }),
        "Swap"     => Some(GateKernel { kernel_name: "swap_kernel",     ptx_file: "swap.ptx", use_theta: false }),
        "T"        => Some(GateKernel { kernel_name: "t_kernel",        ptx_file: "t.ptx", use_theta: false }),
        "TDagger"  => Some(GateKernel { kernel_name: "tdagger_kernel",  ptx_file: "t_dagger.ptx", use_theta: false }),
        "Toffoli"  => Some(GateKernel { kernel_name: "toffoli_kernel",  ptx_file: "toffoli.ptx", use_theta: false }),
        "U2"       => Some(GateKernel { kernel_name: "u2_kernel",       ptx_file: "u2.ptx", use_theta: true  }),
        "U3"       => Some(GateKernel { kernel_name: "u3_kernel",       ptx_file: "u3.ptx", use_theta: true  }),

        _ => None,
    }
}