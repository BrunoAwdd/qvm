# 🟦 `Identity` Gate – QLang Quantum Simulation

The `Identity` gate (I) is a **no-operation (no-op)** quantum gate that leaves the qubit unchanged. It serves as the identity matrix in quantum computing.

---

## 📐 Matrix Representation

```
| 1  0 |
| 0  1 |
```

This gate does not alter the amplitude of the qubit. It's used in circuit design for:
- Synchronization across multiple qubits
- Placeholder operations
- Testing or composite gates

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::identity::Identity;

let id = Identity::new();
let matrix = id.matrix();
```

---

## 🧪 Tests

- `test_identity_matrix_correctness` – Confirms the identity matrix structure.
- `test_identity_name` – Verifies gate name as `"Identity"`.

---

## 📎 Notes

- Does not affect quantum state evolution.
- Useful in control flow, simulation testing, or aligning gate timing.