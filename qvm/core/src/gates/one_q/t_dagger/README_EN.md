# 🟣 `T†` Gate – QLang Quantum Simulation

The `T†` gate (T-dagger) is the **inverse** of the `T` (π/8) gate. It applies a phase shift of −π/4 to the `|1⟩` state and is essential in circuit reversals and quantum error correction.

---

## 📐 Matrix Representation

```
| 1           0 |
| 0   e^(-iπ/4) |
```

Behavior:

- `T†|0⟩ = |0⟩`
- `T†|1⟩ = e^(-iπ/4)|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::t_dagger::TDagger;

let t_dag = TDagger::new();
let matrix = t_dag.matrix();
```

---

## 🧪 Tests

- `test_t_dagger_matrix` – Verifies the correct matrix form with −π/4 phase.
- `test_t_dagger_name` – Confirms the gate name is `"TDagger"`.

---

## 📎 Notes

- Hermitian conjugate of the `T` gate.
- `T† * T = Identity`
- Critical in fault-tolerant and reversible quantum circuit construction.