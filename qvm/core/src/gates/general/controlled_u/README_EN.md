# 🧠 `ControlledU` Gate – QLang Quantum Simulation

The `ControlledU` gate is a **two-qubit quantum gate** that applies an arbitrary unitary matrix to the target qubit **only when the control qubit is in the `|1⟩` state**. It is fundamental in constructing gates like `CNOT`, `CZ`, or even `Controlled-Rx`, `Ry`, etc.

---

## 📐 Matrix Representation

The resulting 4×4 matrix has the structure:

```
| I  0 |
| 0  U |
```

- `I`: 2×2 identity matrix → no operation when control = `|0⟩`
- `U`: 2×2 unitary matrix → applied when control = `|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::controlled_u::ControlledU;

let cu = ControlledU::new_real(0.0, 1.0, 1.0, 0.0); // Acts like a CNOT gate
let matrix = cu.matrix();
```

Or using a complex matrix:

```rust
use ndarray::array;
use qlang::types::qlang_complex::QLangComplex;

let u = array![
    [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
    [QLangComplex::new(0.0, 0.0), QLangComplex::new(-1.0, 0.0)],
];
let cu = ControlledU::new(u, None);
```

---

## 🚀 CUDA Kernel

The project also includes a GPU version of the `ControlledU` gate, implemented in CUDA, for direct manipulation of the quantum state vector.

### Kernel Signature

```cpp
__global__ void controlled_u_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits,
    cuDoubleComplex u00,
    cuDoubleComplex u01,
    cuDoubleComplex u10,
    cuDoubleComplex u11
);
```

This kernel applies `U` to the `target` qubit **only** when the `control` qubit is in the `|1⟩` state. The implementation avoids race conditions by design.

---

## 🧪 Tests

- `test_controlled_u_from_real` – Verifies correct behavior using real values (e.g., simulating CNOT).
- `test_controlled_u_from_matrix` – Ensures the `U` matrix is placed correctly in the 4×4 structure.
- `test_controlled_u_name` – Confirms the gate is correctly identified as `"cu"`.

---

## 📎 Notes

- The provided `U` matrix must be 2×2 and unitary.
- The CUDA kernel assumes `target != control`.
- For larger quantum systems, performance is critical — the kernel was designed with efficiency in mind.
