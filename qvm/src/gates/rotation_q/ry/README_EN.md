# 🔄 `RY(θ)` Gate – QLang Quantum Simulation

The `RY` gate performs a **rotation around the Y-axis** of the Bloch sphere by an angle `θ` (in radians). This is a real-valued single-qubit rotation, frequently used in variational quantum algorithms and quantum neural networks.

---

## 📐 Matrix Representation

```
RY(θ) =
[  cos(θ/2)   -sin(θ/2) ]
[  sin(θ/2)    cos(θ/2) ]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::ry::RY;

let theta = std::f64::consts::PI / 2.0;
let ry = RY::new(theta);
let matrix = ry.matrix();
```

---

## 🧪 Tests

- `test_ry_matrix` – Validates matrix generation for θ = π/2.
- `test_ry_name` – Confirms that the gate name is `"RY"`.

---

## 📎 Notes

- No complex phases involved; all entries are real.
- When `θ = π`, the `RY` gate acts as a Pauli-Y up to a global phase.
- Used to prepare superpositions and adjust amplitudes in quantum circuits.