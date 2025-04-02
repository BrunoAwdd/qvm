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
        "hadamard.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/hadamard.ptx")),
        "rx.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/rx.ptx")),
        "ry.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/ry.ptx")),
        "rz.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/rz.ptx")),
        "pauli_x.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_x.ptx")),
        "pauli_y.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_y.ptx")),
        "pauli_z.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/pauli_z.ptx")),
        "s.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/s.ptx")),
        "t.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/t.ptx")),
        "cnot.ptx" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/cu/ptx/cnot.ptx")),
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
            [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *a, *b
                    )
                ).expect("Falha ao lançar kernel com 3 argumentos");
            }
            [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b), KernelArg::F64(theta)] => {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        *ptr, *a, *b, *theta
                    )
                ).expect("Falha ao lançar kernel com 4 argumentos");
            }
            _ => panic!("Número de argumentos não suportado para kernel {}", kernel_name),
        }
    }

    stream.synchronize().unwrap();
}
