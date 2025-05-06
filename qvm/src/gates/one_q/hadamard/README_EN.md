# ✨ `Hadamard` Gate – QLang Quantum Simulation

The `Hadamard` gate (H) is a fundamental one-qubit quantum gate that creates superposition. When applied to a qubit, it transforms the basis states into equal superpositions:

- `H|0⟩ = (|0⟩ + |1⟩)/√2`
- `H|1⟩ = (|0⟩ - |1⟩)/√2`

---

## 📐 Matrix Representation

The Hadamard gate is defined as:

```
1/sqrt(2) * |  1   1 |
            |  1  -1 |
```

It is both **Hermitian** and **unitary**, making it self-inverse.

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::one_q::hadamard::Hadamard;

let h = Hadamard::new();
let matrix = h.matrix();
```

Apply on a `QVM`:

```rust
let mut qvm = QVM::new(1);
qvm.apply_gate(&Hadamard::new(), 0);
```

---

## 🚀 CUDA Kernel

The Hadamard gate can be applied to a quantum state vector in parallel using the `hadamard_kernel` CUDA kernel:

```cpp
__global__ void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits);
```

- Efficiently applies H to the specified `qubit` in `state`
- Only one thread per entangled pair performs the update to avoid race conditions
- Assumes double precision (cuDoubleComplex) for amplitude representation

---

## 🧪 Tests

- `test_hadamard_matrix` – Confirms the matrix is correct.
- `test_hadamard_name` – Verifies gate identification.
- `test_hadamard_apply` – Applies H and checks probabilistic behavior.
- `test_hadamard_distribution` – Ensures (roughly) 50/50 distribution over many runs.
- `test_measure_many_hadamard` – Integration test with `QLang` commands and multi-qubit measurements.

---

## 📎 Notes

- Core component of quantum algorithms such as Grover's and Shor's.
- Used to initialize qubits into superposition before interference-based computations.