# 🧭 Roadmap — QLang QVM v0.1

This document defines the milestones for QLang version 0.1, including basic quantum gates, support for different backends, and execution through external languages via FFI.

---

## ✅ Supported Gates

### 🔹 1-Qubit Gates

- [x] `x(q)` — Pauli-X
- [x] `y(q)` — Pauli-Y
- [x] `z(q)` — Pauli-Z
- [x] `h(q)` — Hadamard
- [x] `s(q)` — S Phase
- [x] `sdg(q)` — S-dagger
- [x] `t(q)` — T Phase
- [x] `tdg(q)` — T-dagger
- [x] `id(q)` — Identity

### 🔹 Rotation Gates

- [x] `rx(q, θ)`
- [x] `ry(q, θ)`
- [x] `rz(q, θ)`
- [x] `u3(q, θ, φ, λ)`
- [x] `u2(q, φ, λ)`
- [x] `u1(q, λ)`
- [x] `phase(q, θ)`

### 🔹 2-Qubit Gates

- [x] `cnot(c, t)`
- [x] `swap(q1, q2)`
- [x] `iswap(q1, q2)`
- [x] `cz(c, t)`
- [x] `cy(c, t)`

### 🔹 3-Qubit Gates

- [x] `toffoli(c1, c2, t)`
- [x] `fredkin(c, q1, q2)`

### 🔹 Measurements

- [x] `m(q)` or `measure_all()`
- [ ] `mx(q)` — _(planned as helper using `h(q)` + `m(q)`)_
- [ ] `my(q)` — _(planned as helper using `sdg(q)` + `h(q)` + `m(q)`)_

---

## ✅ Backends

- [x] CPU (via `ndarray`, multi-threaded with `rayon`)
- [x] CUDA (via `cust` and dedicated PTX kernels)
- [x] Batch execution and parallelism (`BatchRunner`)

---

## 🚧 v0.1 Scope

| Item                                     | Status       |
| ---------------------------------------- | ------------ |
| QLang language and AST                   | ✅ Completed |
| Execution from `.ql` files               | ✅ Completed |
| Inline interpretation (string)           | ✅ Completed |
| Parametrized `U3`, `U2` gate support     | ✅ Completed |
| Parallel job execution (`BatchRunner`)   | ✅ Completed |
| Gates `iswap`, `cz`, `cy`, `u1`, `phase` | ✅ Completed |
| Helpers `mx`, `my` (X/Y measurements)    | ✅ Completed |
| Library packaging as `libQLang.so`       | ✅ Completed |
| Python bindings (via `ctypes`)           | ✅ Completed |

---

## 🧪 Out of Scope for v0.1

| Feature                               | Planned for |
| ------------------------------------- | ----------- |
| Noise model                           | 🔜 v0.2     |
| Circuit optimization / gate fusion    | 🔜 v0.2     |
| `wgpu` (WebGPU) backend               | 🔮 Future   |
| State slicing or tensor ops execution | 🔜 v0.3     |

---

## ✅ Backends

- [x] CPU
- [x] CUDA
- [ ] WGPU (WebGPU)

## 🧠 Version 0.2 - Circuit Batching (CPU and CUDA)

### 1. Small Circuit Batching (CPU and CUDA)

- [x] Abstraction of `CircuitJob` (circuit + state)
- [x] Batch execution using `Vec<CircuitJob>`
- [x] Parallel execution with `rayon` (CPU)
- [x] Sequential CUDA execution (1 kernel per circuit)

#### 🔧 Challenges:

- CUDA does not handle heterogeneous launches well
- Buffer management for `Vec<State>` in GPU
- Synchronization between different kernels or grids

## 🧠 Version 0.3 - Tensor Networks and Entanglement Estimation

### 2. Tensor Networks (CPU) v0.3

> Simulation based on tensor networks (like MPS), ideal for circuits with low entanglement. Enables +40 qubit simulations without exhausting RAM.

- [x] Study and selection of model (MPS as initial base)
- [ ] Abstraction of `TensorNode` and `TensorNetwork`
- [x] Representation of states as chained tensors
- [x] Application of gates as tensor contractions
- [x] Full support for 1-qubit gates
- [x] Support for 2-qubit and 3-qubit gates
- [x] Simulation of measurements and local collapse
- [x] Backend flag: `--backend=tensor`
- [ ] Benchmark: `tensor vs full-state` in time and RAM usage
- [ ] Create `estimate_entanglement(&Circuit)`
- [ ] Measure-X and Measure-Y
- [ ] Fix CUDA Backend. Device should never to be dropped, but we can play with context and stream.

## 🔬 Version 0.4 — Scientific Expansion and Interoperability

### 🧪 1. Noise Modeling

- [ ] Implement **depolarizing** noise
- [ ] Implement **bit-flip** noise
- [ ] Implement **phase-flip** noise
- [ ] Allow enabling/disabling noise via QVM parameter
- [ ] Document the probabilistic model used
- [ ] Write tests to validate statistical distribution
- 👉 Simulate realistic environments and test fault tolerance

### 🔗 2. QASM Export and Qiskit Integration

- [ ] Export QLang circuits in `.qasm` format
- [ ] Add `to_qasm()` method in QLang AST
- [ ] Implement initial parser to import `.qasm` (v1.0)
- [ ] Add practical example of interoperability with Qiskit
- 👉 Opens integration with real pipelines and external compilers

# 🧭 Roadmap — QLang QVM v0.5

Version 0.5 focuses on **usability, discoverability, and practical performance**. No new gates, but expands language and interface power.

---

## ✨ Planned Features

### ✅ Circuit Optimization

- [ ] Remove redundant gates (e.g., `x; x = id`)
- [ ] Local simplifications (e.g., `h; z; h = x`)
- [ ] Merge consecutive gates on same qubit

> 🎯 **Goal:** reduce computational cost without changing output.

### ✅ REPL (Interactive Terminal)

- [ ] Interactive prompt with history and autocomplete
- [ ] Line-by-line command evaluation
- [ ] Partial visualization (per qubit or gate)

> 🎯 **Goal:** test and learn QLang quickly, no files needed.

### ✅ Visualization

- [ ] ASCII rendering of circuits
- [ ] Clear state vector display (`|00⟩: 0.707 + 0.0i`)
- [ ] Future integration with `matplotlib` or `plotly` via Python

> 🎯 **Goal:** assist in education, debugging, and analysis.

### ✅ Batch Execution

- [ ] CUDA kernel to operate multiple state vectors in **one launch**
- [ ] Benchmark: `batch vs single` in time and resource usage

## 🧪 Possible Extras (if time permits)

- [ ] Extra REPL commands (`.save`, `.load`, `.exit`)
- [ ] VSCode plugin or TUI interface

# 🌐 Roadmap — QLang QVM v0.6 (Web Support)

Version 0.6 brings QLang to the web, making the simulator accessible in browsers and online platforms.

---

## 🚀 Web and WASM Support

### ✅ WebAssembly (WASM)

- [ ] Compile QVM core to WASM (`wasm32-unknown-unknown`)
- [ ] Expose minimal interface via JavaScript (create, apply_gate, measure)
- [ ] Adapt memory management for browser use

> 🎯 **Goal:** run QLang directly in HTML pages.

---

### ✅ Online Playground

- [ ] Web interface with `.ql` code editor
- [ ] "Run" button that executes in-browser (WASM)
- [ ] Display final state and measurements

> 🎯 **Goal:** test circuits online with zero install.

---

### ✅ In-Browser Visualization

- [ ] Graphical circuit representation (JS/CSS/SVG)
- [ ] Live display of state vector
- [ ] Export options: `.ql`, `.qasm`, `.json`

> 🎯 **Goal:** make QLang more visual and accessible for teaching and demo.

---

## 🔗 Future Integrations

- [ ] Jupyter Notebook plugin via `pyodide`
- [ ] Export to educational sites (e.g., Khan Academy, Observable)
- [ ] NPM module (`@qlang/qvm`) for use with React, Vue, etc.

---

## 📦 Expected Result

A WebAssembly-compiled version accessible from any modern browser, enabling real-time QLang circuit creation, execution, and visualization — with no dependencies.

---

## 🛠️ Future Ideas

- [ ] Stabilizers (Clifford-only)
- [ ] Realistic noise (bit-flip, depolarization)
- [ ] QASM export
- [ ] Qiskit integration
- [ ] Web/CLI interactive interface
- [ ] Bindings for Go, Julia, etc.

---

# 🧭 QLang Roadmap — v0.7 to v1.0

## ✅ v0.7 — Standard Quantum Algorithms

> Introduces a set of classic quantum algorithms as reusable, ready-to-run circuits.

- [ ] Implement circuit library for:
- [ ] Grover Search
- [ ] QFT (Quantum Fourier Transform)
- [ ] Deutsch-Jozsa
- [ ] Bernstein-Vazirani
- [ ] Bell State
- [ ] GHZ State
- [ ] Internal `CircuitLibrary` structure
- [ ] Helper functions (`qlib.grover(n)`, etc.)
- [ ] Examples in README and `.ql` scripts
- [ ] Unit tests to validate all algorithms

---

## ✅ v0.8 — Optimizer and Transpiler

> Circuit performance improvements with smart simplification and rewrite rules.

- [ ] Circuit transpiler:
- [ ] Merge consecutive gates
- [ ] Cancel inverse gates (`x; x = id`)
- [ ] Simplify via identities (`h; z; h = x`)
- [ ] Optimizer with configurable heuristics
- [ ] Transformation logger (`--opt-log`)
- [ ] CLI flag `--optimize`
- [ ] Export `.qopt` files with optimized circuits

---

## ✅ v0.9 — Advanced Noise + Physical Backends

> Realistic simulations and export to hardware-compatible formats.

- [ ] Noise simulator:
- [ ] Channels: depolarizing, bit-flip, phase-flip
- [ ] Custom noise per gate
- [ ] Noise profiles (`--noise=ibm_standard`)
- [ ] Export to QASM v2.0
- [ ] Optional Qiskit Cloud integration (`--export-to=qiskit`)
- [ ] `qlang-to-qiskit` tool for direct interoperability

---

## ✅ v1.0 — Stability, Interactivity and Ecosystem

> The official stable release with strong documentation and full tooling.

- [ ] **CLI / REPL final version**
- [ ] Autocomplete, highlight, macros
- [ ] History, `.save`, `.load`, `.viz`
- [ ] **Web Playground 1.0**
- [ ] Online `.ql` editor + WASM runner
- [ ] Graphical circuit viewer
- [ ] **Full Documentation**
- [ ] `docs.qlang.dev` (or similar)
- [ ] Tutorials for Python, C, JS
- [ ] **Official Packages**
- [ ] `pip install qlang`
- [ ] `npm install @qlang/qvm`
- [ ] `cargo add qlang`
- [ ] **Smart multi-backend support**
- [ ] Automatic: CPU / GPU / Tensor / WASM
- [ ] **Final `.ql` language version**
- [ ] Stable syntax with robust parser
- [ ] **Validation with IBM circuits**
- [ ] 50+ real-world circuit benchmarks

---

## 📦 Post-1.0 (Future Ideas)

- [ ] Official VSCode plugin (syntax, execution)
- [ ] QLang-based education framework
- [ ] Quantum Error Correction support
- [ ] Physical hardware emulation (latency, decoherence)
- [ ] WebGPU backend (via `wgpu`)
- [ ] Visual DSL for drag-and-drop circuit design

## 📜 License and Repository

- Repository: `Coming soon`
- License: MIT or Apache 2.0
- Mission: Make QLang an accessible, powerful, and extensible simulator for research and education.
