# QLang — Language Roadmap

> **Core philosophy:** The simulator is infrastructure. The _language_ is the product.
> QLang's differentiator is being the first practical quantum language that sits
> **above circuits** — with real control flow, first-class functions, quantum
> conditionals (`qif`), and a type system that enforces physical laws at compile time.

---

## Status legend

| Symbol | Meaning                            |
| ------ | ---------------------------------- |
| ✅     | Done                               |
| 🔨     | In progress / landed in this cycle |
| 🔲     | Planned                            |

---

## v0.1 — Simulator foundation ✅

- State vector simulator (CPU via Rayon)
- CUDA backend with correctness-first device/host fallback; native PTX kernels remain experimental
- Exact tensor/MPS backend for small circuits (no truncation yet)
- 21 quantum gates: Pauli, Hadamard, S/T, Rx/Ry/Rz, U1/U2/U3, CNOT, SWAP,
  iSWAP, CY, CZ, Toffoli, Fredkin, ControlledU
- Batch execution (`BatchRunner`)
- Stable C ABI and typed Python `ctypes` wrapper (local build/install)
- `.ql` file execution + inline interpretation
- Cross-platform C-compatible FFI (`.so`, `.dylib`, `.dll`)

---

## v0.2 — Language completion ✅

_Goal: QLang must be a complete imperative language with quantum semantics._

- ✅ `fn` / user-defined functions with parameters
- ✅ `let` / `assign` — classical variables
- ✅ `if` / `else` — classical conditional
- ✅ `while` / `for` — classical loops
- ✅ `qif` / `qif-else` — quantum conditional (superposition-safe, no collapse)
- ✅ Arrays and indexing
- ✅ `import` — load `.ql` files and merge function definitions
- ✅ `return expr` — functions return values
- ✅ `let x = fn(args)` — function calls as expressions
- ✅ `break` / `continue` — loop control
- ✅ Proper error types (`QLangError`) — no more silent `println!` failures
- 🔲 `return` without value (void functions)
- 🔲 Error messages with source line numbers

---

## v0.3 — Type system ✅

_Goal: the compiler rejects programs that are physically impossible._

- ✅ `QLangType` — `qubit`, `bit`, `int`, `float`, `bool`, `[T]`, `void`
- ✅ Linear type for `qubit`: `measure(q)` consumes `q`; using `q` afterwards
  is a **compile-time error** (no-cloning theorem enforced statically)
- ✅ Gate-on-measured-qubit detection
- ✅ `qif` body validation: measurements inside `qif` collapse the superposition
  that `qif` is supposed to preserve — this is now a compile-time error
- ✅ Out-of-range qubit index detection
- ✅ Type annotations: `let q: qubit = alloc(0)`, `fn bell(q1: qubit, q2: qubit) -> void`
- ✅ `alloc(n)` — returns `Value::Qubit(n)` (first-class qubit reference)
- ✅ `measure(q)` — returns `Value::Bit(u8)` (typed classical result)
- ✅ No-cloning detection: `let q2 = q1` marks `q1` as moved; subsequent use is a compile-time error
- 🔲 Full type inference — no annotation required in all cases
- 🔲 Error messages with source line numbers

---

## v0.4 — Quantum semantics

_Goal: first-class quantum resource management._

- `q2 = q1` as programmable entanglement — creates a Bell pair at the language level
  (physically valid; distinct from copying which violates no-cloning)
- `with anc = alloc(1) { ... }` — scoped ancilla allocation; compiler
  guarantees `anc` is in |0⟩ on entry and is deallocated on exit
- Qubit lifetime tracking — the compiler knows when each qubit "dies"
- `qif` type-checks that its body is unitary (no measurement, no ancilla leak)
- Named qubit registers: `let q = qubits(3)` instead of implicit indices

---

## v0.5 — Tooling

_Goal: the language is usable beyond a research prototype._

- REPL with history and tab completion
- VSCode syntax highlighting extension (`.ql` grammar)
- LSP skeleton (go-to-definition for functions, hover types)
- `std.ql` expanded: Grover's algorithm, QFT, Deutsch-Jozsa, phase kickback
- Compiler error messages with file + line number

---

## v0.6 — Simulator performance

_The simulator catches up to the language._

- ✅ Exact tensor/MPS backend with CPU parity tests
- Tensor truncation and contraction planning for large low-entanglement circuits
- Circuit optimizer: cancel redundant gate pairs (e.g. `H H = I`, `X X = I`)
- Noise modeling: depolarizing, bit-flip, phase-flip channels
- 40+ qubit simulation via tensor contraction

---

## v0.7 — Interoperability

- ✅ Shared parser compiled to WASM and consumed by the browser viewer
- QASM 2.0 import/export (bridge to IBM Quantum hardware)
- Python wheels and type-annotated stubs (source package and runtime wrapper already present)
- npm package (`@qlang/core`) for JavaScript/TypeScript

---

## v1.0 — Production language

- Module / package system: `import qlang.algorithms.grover`
- Full LSP (Language Server Protocol) for any editor
- Comprehensive documentation site
- Algorithm library: Grover, Shor (partial), HHL, QAOA primitives
- Benchmarks vs Qiskit Aer, Cirq — publish results
- Official Cargo crate, PyPI package, npm package

---

## Post-1.0 — Research directions

- **Automatic uncomputation** (inspired by Silq): ancilla qubits freed from
  scope are automatically uncomputed without measurement, eliminating
  "garbage" that corrupts superposition.
- **Hardware transpiler**: map QLang circuits to IBM / IonQ / Rigetti native
  gate sets.
- **Quantum-classical hybrid**: first-class variational circuits (VQE, QAOA)
  with classical optimizer loop.
