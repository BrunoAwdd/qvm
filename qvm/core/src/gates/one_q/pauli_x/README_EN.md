# ❌ `Pauli-X` Gate – QLang Quantum Simulation

The `Pauli-X` gate (also known as the `X` gate or quantum NOT gate) flips the state of a single qubit. It is one of the fundamental Pauli gates used in quantum computation.

---

## 📐 Matrix Representation

```
| 0  1 |
| 1  0 |
```

This gate swaps the amplitude between |0⟩ and |1⟩:

- `X|0⟩ = |1⟩`
- `X|1⟩ = |0⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::pauli_x::PauliX;

let x = PauliX::new();
let matrix = x.matrix();
```

---

## 🧪 Tests

- `test_pauli_x_matrix` – Validates the matrix structure of the Pauli-X gate.
- `test_pauli_x_name` – Confirms the gate is named `"PauliX"`.

---

## 📎 Notes

- Equivalent to a classical NOT gate in behavior.
- Commonly used for bit flips and constructing more complex operations like CNOT.