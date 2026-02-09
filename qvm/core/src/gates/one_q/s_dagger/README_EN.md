# 🔷 `S†` Gate – QLang Quantum Simulation

The `S†` gate (S-dagger) is the **inverse** of the `S` phase gate. It applies a negative π/2 phase (−i) to the `|1⟩` state and is a common component in quantum circuit reversals and corrections.

---

## 📐 Matrix Representation

```
| 1    0 |
| 0  -i |
```

Behavior:

- `S†|0⟩ = |0⟩`
- `S†|1⟩ = -i|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::s_dagger::SDagger;

let s_dag = SDagger::new();
let matrix = s_dag.matrix();
```

---

## 🧪 Tests

- `test_s_dagger_matrix` – Validates the matrix implementation.
- `test_s_dagger_name` – Verifies the gate's name is `"SDagger"`.

---

## 📎 Notes

- `S† * S = Identity`
- Used in reversing quantum operations and error correction.