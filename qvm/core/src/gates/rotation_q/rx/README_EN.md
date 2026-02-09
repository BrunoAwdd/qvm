# 🔄 `RX(θ)` Gate – QLang Quantum Simulation

The `RX` gate performs a **rotation around the X-axis** of the Bloch sphere by an angle `θ` (in radians). It is a parametric, single-qubit gate used in many quantum algorithms and variational circuits.

---

## 📐 Matrix Representation

```
RX(θ) = cos(θ/2)·I − i·sin(θ/2)·X

      =
      [ cos(θ/2)    -i·sin(θ/2) ]
      [ -i·sin(θ/2)  cos(θ/2)   ]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::rx::RX;

let theta = std::f64::consts::PI / 2.0;
let rx = RX::new(theta);
let matrix = rx.matrix();
```

---

## 🧪 Tests

- `test_rx_gate_matrix_pi_2` – Validates the matrix for θ = π/2.
- `test_rx_gate_name` – Confirms the gate name is `"RX"`.

---

## 📎 Notes

- This is a parametric gate (depends on θ).
- Common in quantum neural networks and variational algorithms.
- When `θ = π`, the RX gate behaves like a Pauli-X.