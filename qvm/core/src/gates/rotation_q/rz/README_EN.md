# 🌀 `RZ(θ)` Gate – QLang Quantum Simulation

The `RZ` gate performs a **rotation around the Z-axis** of the Bloch sphere by an angle `θ` (in radians). It applies opposite complex phases to the `|0⟩` and `|1⟩` components.

---

## 📐 Matrix Representation

```
RZ(θ) =
[ e^{-iθ/2}      0      ]
[     0      e^{iθ/2}   ]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::rz::RZ;

let theta = std::f64::consts::PI / 2.0;
let rz = RZ::new(theta);
let matrix = rz.matrix();
```

---

## 🧪 Tests (suggested)

- Validate the generated matrix has the correct phases ±θ/2.
- Confirm the gate name is `"RZ"`.

---

## 📎 Notes

- The `RZ` gate is diagonal and purely phase-based.
- It is often used in combination with `RX` and `RY` to implement arbitrary single-qubit rotations.
- When `θ = π`, the `RZ` gate acts like a Pauli-Z up to a global phase.