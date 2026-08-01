# QLang

[![CI](https://github.com/BrunoAwdd/qvm/actions/workflows/ci.yml/badge.svg)](https://github.com/BrunoAwdd/qvm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

QLang is an experimental typed quantum language and simulator written in Rust. The active implementation is a Cargo workspace under `qvm/` with a shared syntax crate, CPU/CUDA/tensor backends, a command-line interface, C ABI, Python wrapper, and a React viewer powered by WebAssembly.

## Current status

- CPU is the default backend and is covered by amplitude-level Bell, GHZ, Toffoli, and Fredkin tests.
- Tensor uses an exact MPS decomposition without truncation. It prioritizes correctness; optimization and bond-dimension limits remain future work.
- CUDA uses a correctness-first host/device fallback by default. Existing PTX kernels are available through the experimental `cuda-kernels` feature and are not yet part of the trusted path.
- The type checker validates annotations, assignments, arrays, function signatures and returns, control-flow context, and conservative linear ownership of qubits.
- The viewer uses the same `qlang-syntax` parser as the native runtime through WASM.

This is still a pre-1.0 project. Noise modeling, an optimizer, a complete module system, and validated accelerated CUDA kernels are not implemented.

## Workspace

| Crate | Purpose |
| --- | --- |
| `qlang_core` | Complex numbers, states, tensor networks, and gate matrices |
| `qvm` | CPU, CUDA, and tensor execution backends |
| `qlang-syntax` | Shared lexer, parser, AST, aliases, and syntax errors |
| `qlang` | Interpreter, type checker, batch execution, and CLI |
| `qlang-ffi` | Stable C ABI built as a `cdylib` |
| `qlang-wasm` | WASM projection of the shared parser for the viewer |

## CLI

From the repository root:

```bash
cd qvm
cargo run -p qlang -- check examples/teleportation.ql
cargo run -p qlang -- run examples/teleportation.ql
```

Additional tracked examples include `bell.ql`, `ghz.ql`, and an intentionally invalid `type_error.ql` for demonstrating static diagnostics.

`portfolio_hedge.ql` sketches a finance-shaped use case for `qif`: a hedge decision that reacts to a market-regime qubit before it is measured, so the branch stays coherent instead of collapsing early. `portfolio_hedge_reuse_error.ql` is the intentionally invalid counterpart — it reuses a qubit after `measure`, which the linear type checker rejects at compile time.

QLang source example:

```qlang
create(2)

fn bell(q0: qubit, q1: qubit) -> void {
    h(q0)
    cnot(q0, q1)
}

let q0: qubit = alloc(0)
let q1: qubit = alloc(1)
bell(q0, q1)
```

## Backends

```bash
cd qvm

# CPU
cargo test --workspace

# Exact tensor/MPS backend
cargo test --workspace --features tensor

# CUDA compile check
cargo check -p qvm --features cuda

# CUDA runtime tests require an NVIDIA GPU
cargo test -p qvm --features cuda --test quantum_circuits -- --test-threads=1
```

Do not enable `cuda-kernels` for correctness-sensitive work yet. That feature exists to validate and repair individual PTX kernels.

## Python and C ABI

Build and install the CPU binding:

```bash
cd qvm
cargo build -p qlang-ffi --release
cd ..
python -m pip install -e .
```

```python
from qlang import QLang

runtime = QLang(2)
runtime.run("h(0)\ncnot(0, 1)")
print(runtime.state())
print(runtime.measure_all())
```

Set `QLANG_LIBRARY` when the shared library is outside `qvm/target/{debug,release}`. The ABI uses caller-owned measurement buffers and provides `qlang_last_error()` for diagnostics.

## Viewer/WASM

```bash
cd qvm
wasm-pack build qlang-wasm --target web \
  --out-dir ../../qlang-viewer/src/wasm --out-name qlang_wasm
cd ../qlang-viewer
pnpm install
pnpm dev
```

The viewer is a static circuit projection, not a browser simulator. Dynamic variables and control flow produce explicit diagnostics rather than being silently interpreted by a regex parser.

## Verification

```bash
cd qvm
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo test --workspace --features tensor

cd ../qlang-viewer
pnpm build
pnpm lint

cd ..
PYTHONPATH=python python -m pytest tests/python/test_ffi_runtime.py -q
```

GitHub Actions runs the CPU workspace on Linux, macOS, and Windows; tensor, CUDA compilation, WASM/viewer, and Python/FFI have dedicated jobs. CUDA runtime tests are available as a manually triggered workflow for a self-hosted GPU runner.

See [ROADMAP.md](ROADMAP.md) for planned work.

## License

QLang is licensed under the [Apache License 2.0](LICENSE).
