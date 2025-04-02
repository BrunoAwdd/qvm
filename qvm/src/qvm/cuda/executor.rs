#![cfg(feature = "cuda")]

use cust::{
    memory::{DeviceBuffer, DevicePointer},
    module::Module,
    prelude::*,
    stream::Stream,
    context::Context,
};

use crate::qvm::cuda::types::CudaComplex;

/// Representa um argumento possível para um kernel CUDA
pub enum KernelArg {
    Ptr(DevicePointer<CudaComplex>),
    I32(i32),
    F64(f64),
}

/// Lança um kernel CUDA com até 4 argumentos
pub fn launch_cuda_gate_kernel(
    kernel_name: &str,
    ptx_filename: &str,
    args: &[KernelArg],
    stream: &Stream,
    _context: &Context,
) {

    let ptx_code = match ptx_filename {
        "cnot.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/cnot.ptx")),
        "fredkin.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/fredkin.ptx")),
        "hadamard.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/hadamard.ptx")),
        "rx.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/rx.ptx")),
        "ry.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/ry.ptx")),
        "rz.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/rz.ptx")),
        "pauli_x.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_x.ptx")),
        "pauli_y.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_y.ptx")),
        "pauli_z.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_z.ptx")),
        "s.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/s.ptx")),
        "swap.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/swap.ptx")),
        "t.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/t.ptx")),
        "toffoli.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/toffoli.ptx")),
        "u3.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/u3.ptx")),
        _ => panic!("PTX desconhecido: {}", ptx_filename),
    };


    let module = Module::from_ptx(ptx_code, &[]).unwrap();
    let function = module.get_function(kernel_name).unwrap();

    // Estimativa de tamanho com base no maior valor de qubit
    let size = 1 << args.iter().filter_map(|arg| match arg {
        KernelArg::I32(v) => Some(*v as usize), 
        _ => None,
    }).max().unwrap_or(8);


    let block_size = 256;
    let grid_size = ((size + block_size - 1) / block_size) as u32;

    unsafe {
        match args {
            // Ex: CNOT, SWAP, etc.
            [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *a, *b
                    )
                ).expect("Falha ao lançar kernel com 3 argumentos");
            }

            // Ex: RX, RY, RZ
            [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b), KernelArg::F64(theta)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *a, *b, *theta
                    )
                ).expect("Falha ao lançar kernel com 4 argumentos");
            }

            // Ex: Toffoli
            [KernelArg::Ptr(ptr), KernelArg::I32(q0), KernelArg::I32(q1), KernelArg::I32(q2), KernelArg::I32(n)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *q0, *q1, *q2, *n
                    )
                ).expect("Falha ao lançar kernel com 5 argumentos");
            }

            // ✅ Novo caso: U3
            [KernelArg::Ptr(ptr), KernelArg::I32(qubit), KernelArg::I32(n), KernelArg::F64(theta), KernelArg::F64(phi), KernelArg::F64(lambda)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *qubit, *n, *theta, *phi, *lambda
                    )
                ).expect("Falha ao lançar kernel com 6 argumentos (U3)");
            }

            _ => panic!("Número de argumentos não suportado para kernel {}", kernel_name),
        }
    }


    stream.synchronize().unwrap();
}
