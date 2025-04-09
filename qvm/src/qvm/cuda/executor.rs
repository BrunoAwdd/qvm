#![cfg(feature = "cuda")]

use cust::{
    memory::{DeviceBuffer, DevicePointer},
    module::Module,
    prelude::*,
    stream::Stream,
    context::Context,
};

use crate::types::qlang_complex::QLangComplex;

/// Representa um argumento possível para um kernel CUDA
pub enum KernelArg {
    Ptr(DevicePointer<QLangComplex>),
    I32(i32),
    F64(f64),
}

pub fn launch_cuda_gate_kernel(
    kernel_name: &str,
    ptx_filename: &str,
    args: &[KernelArg],
    stream: &Stream,
    context: &Context,
) {
    let ptx_code = load_ptx(ptx_filename);
    let module = Module::from_ptx(ptx_code, &[]).unwrap();
    let function = module.get_function(kernel_name).unwrap();

    let grid_size = estimate_grid_size(args);

    unsafe {
        launch_kernel_args(&function, stream, grid_size, args);
    }

    stream.synchronize().unwrap();
}


fn load_ptx(ptx_filename: &str) -> &'static str {
    match ptx_filename {
        // General Gates
        "contolled_u.ptx"   => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/general/controlled_u/controlled_u.ptx")),
        // One-Qubit Gates
        "hadamard.ptx"      => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/hadamard/hadamard.ptx")),
        "pauli_x.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/pauli_x/pauli_x.ptx")),
        "pauli_y.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/pauli_y/pauli_y.ptx")),
        "pauli_z.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/pauli_z/pauli_z.ptx")),
        "s.ptx"             => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/s/s.ptx")),
        "s_dagger.ptx"      => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/s_dagger/s_dagger.ptx")),
        "t.ptx"             => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/t/t.ptx")),
        "t_dagger.ptx"      => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/one_q/t_dagger/t_dagger.ptx")),
        // Rotation Gates
        "rx.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/rx/rx.ptx")),
        "ry.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/ry/ry.ptx")),
        "rz.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/rz/rz.ptx")),
        "phase.ptx"    => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/phase/phase.ptx")),
        "u1.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/u1/u1.ptx")),
        "u2.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/u2/u2.ptx")),
        "u3.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/rotation_q/u3/u3.ptx")),
        // Two-Qubit Gates
        "cnot.ptx"     => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/two_q/cnot/cnot.ptx")),
        "cy.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/two_q/cy/cy.ptx")),
        "cz.ptx"       => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/two_q/cz/cz.ptx")),
        "swap.ptx"     => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/two_q/swap/swap.ptx")),
        // Three-Qubit Gates
        "fredkin.ptx"  => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/three_q/fredkin/fredkin.ptx")),
        "toffoli.ptx"  => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gates/three_q/toffoli/toffoli.ptx")),
        _ => panic!("PTX desconhecido: {}", ptx_filename),
    }
}

fn estimate_grid_size(args: &[KernelArg]) -> u32 {
    let max_index = args.iter().filter_map(|arg| match arg {
        KernelArg::I32(v) => Some(*v as usize),
        _ => None,
    }).max().unwrap_or(8);

    let block_size = 256;
    ((1 << max_index) + block_size - 1) / block_size
}

unsafe fn launch_kernel_args(
    function: &Function,
    stream: &Stream,
    grid_size: u32,
    args: &[KernelArg],
) {
    match args {
        [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b)] => {
            cust::launch!(function<<<grid_size, 256, 0, stream>>>(*ptr, *a, *b))
                .expect("Kernel 3 args");
        }
        [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b), KernelArg::F64(theta)] => {
            cust::launch!(function<<<grid_size, 256, 0, stream>>>(*ptr, *a, *b, *theta))
                .expect("Kernel 4 args");
        }
        [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b), KernelArg::I32(c), KernelArg::I32(d)] => {
            cust::launch!(function<<<grid_size, 256, 0, stream>>>(*ptr, *a, *b, *c, *d))
                .expect("Kernel 5 args");
        }
        [KernelArg::Ptr(ptr), KernelArg::I32(a), KernelArg::I32(b), KernelArg::F64(t), KernelArg::F64(p), KernelArg::F64(l)] => {
            cust::launch!(function<<<grid_size, 256, 0, stream>>>(*ptr, *a, *b, *t, *p, *l))
                .expect("Kernel 6 args");
        }
        _ => panic!("Número de argumentos não suportado"),
    }
}
