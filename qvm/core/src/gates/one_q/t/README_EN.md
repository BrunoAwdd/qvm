# 🟣 `T` Gate – QLang Quantum Simulation

The `T` gate (π/8 gate) is a **single-qubit phase gate** that applies a π/4 phase shift to the `|1⟩` state. It is a non-Clifford gate and essential for universal quantum computation.

---

## 📐 Matrix Representation

```
| 1         0 |
| 0  e^(iπ/4) |
```

Behavior:

- `T|0⟩ = |0⟩`
- `T|1⟩ = e^(iπ/4)|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::t::T;

let t = T::new();
let matrix = t.matrix();
```

---

## 🧪 Tests (suggested)

- Ensure the matrix matches the expected complex phase form.
- Confirm that the gate name is `"T"`.

---

## 📎 Notes

- `T * T = S`
- Inverse: `T†` (−π/4 phase)
- Enables universality when combined with Clifford gates (H, S, CNOT).