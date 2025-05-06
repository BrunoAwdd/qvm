# 🔺 `Phase(θ)` Gate – QLang Quantum Simulation

The `Phase` gate is a **generalized single-qubit phase gate** that applies a complex phase `θ` (in radians) to the `|1⟩` state. It subsumes the behavior of S, T, and T† gates as special cases.

---

## 📐 Matrix Representation

```
| 1       0 |
| 0   e^{iθ} |
```

Behavior:

- `Phase(θ)|0⟩ = |0⟩`
- `Phase(θ)|1⟩ = e^{iθ}|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::phase::Phase;

let theta = std::f64::consts::PI / 2.0;
let p = Phase::new(theta);
let matrix = p.matrix();
```

---

## ⛓️ Special Cases

- `Phase(π/2)` → `S` gate
- `Phase(π/4)` → `T` gate
- `Phase(−π/4)` → `T†` gate

---

## 🧪 Tests

- `test_phase_gate_matrix_theta_pi_2` – Validates the matrix for `θ = π/2`.
- `test_phase_gate_name` – Verifies the gate is named `"phase"`.

---

## 📎 Notes

- The phase is applied to the `|1⟩` state only.
- Used in algorithms requiring variable phase control.