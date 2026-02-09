# 🌀 `Pauli-Y` Gate – QLang Quantum Simulation

The `Pauli-Y` gate is a fundamental **single-qubit quantum gate** that flips the qubit like `Pauli-X`, but also applies a complex phase (`±i`). It is essential in quantum algorithms involving phase-sensitive rotations.

---

## 📐 Matrix Representation

```
|  0  -i |
|  i   0 |
```

When applied to basis states:

- `Y|0⟩ = i|1⟩`
- `Y|1⟩ = -i|0⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::pauli_y::PauliY;

let y = PauliY::new();
let matrix = y.matrix();
```

---

## 🚀 CUDA Kernel

```cpp
__global__ void pauli_y_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
);
```

- Applies `Y` with manual complex multiplication by ±i for performance.
- Only `i < partner` threads perform swaps to avoid duplicate computation.

---

## 🧪 Tests

- `test_pauli_y_matrix` – Validates that the matrix matches the theoretical Pauli-Y.
- `test_pauli_y_name` – Checks that the gate is labeled as `"pauliY"`.

---

## 📎 Notes

- Introduces both bit and phase flips.
- Important in quantum operations requiring complex phase manipulation.