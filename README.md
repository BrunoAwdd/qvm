# 🧠 QLang — Modern Quantum Simulator in Rust (CPU + GPU)

**QLang** is a quantum language and simulator designed to be lightweight, accessible, and incredibly powerful. Written in Rust, it combines high performance with a simple interface using `.ql` scripts, supports multiple programming languages (Python, C, JavaScript...), and features true CPU and CUDA backends.

> "Simulate 30+ qubits effortlessly — with clean and modular code."

---

## ⚡ Key Features

- 🧩 **Custom language (QLang)**: intuitive and minimal syntax
- 🧠 **Full state vector simulator**
- 🚀 **Parallel backends**: `CPU (rayon)` and `GPU (CUDA)`
- 🔁 **Batch execution** with `BatchRunner`
- 🔧 **Complete quantum gates** (Hadamard, Pauli, U3, Toffoli...)
- 📦 **Bindings for Python, C, JavaScript, Ruby, and more**
- 🛠️ **Gate-specific automated tests**
- 🔭 **Structured roadmap up to version 0.6** (Tensor, WASM, Qiskit, REPL...)

---

## 🛠️ Quick Installation

### 🐍 Via Python (pip)

```bash
pip install qlang
```

## 💻 Binaries

### You can also download prebuilt binaries (`.so` / `.dll`) to use with:

- C/C++
- Ruby
- JavaScript (via `ffi-napi`)
- Java (via JNI)

Binaries available for: **Linux, Windows, macOS, and CUDA**.

## Exemples

### Python (.py)

```python
from qlang import QLangScript
from math import pi

q = QLangScript("cpu")
q.create(2)
q.h(0)
q.cnot(0, 1)
q.m()
q.run()
print("Resultado:", q.get_measurement_result())
```

### Qlang (.ql)

```
create(2)
h(0)
cnot(0,1)
m()
```

`bash cargo run -- my_circuit.ql`

## 🧭 Roadmap

QLang is evolving through well-defined versions:

- ✅ v0.1 — Gates, batching, CUDA, CLI, Python bindings
- 🔜 v0.2 — Optimizations, benchmarking, QASM export
- 🔜 v0.3 — Tensor Networks and slicing
- 🔜 v0.4 — Noise modeling, Qiskit integration
- 🔜 v0.5 — REPL, visualization, automatic simplifications
- 🔜 v0.6 — Web support via WASM

See the full `ROADMAP.md` for details.

---

## 🤝 Contribute

We welcome contributions in areas such as:

- New gates
- Test coverage
- Alternative backends
- Visualization tools
- Web interface
- Translations and documentation

---

## 📜 License

Open-source under the MIT or Apache 2.0 license.

---

## Author

Developed with curiosity and care by **Bruno Oliveira**.

> "Simulating the future is easier than it seems. All you need is a bit of QLang."
