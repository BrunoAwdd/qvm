# 🔒 `Controlled-Z (CZ)` Gate – QLang Quantum Simulation

The `CZ` gate is a **two-qubit controlled-Z gate**. It applies a Pauli-Z to the target qubit **only when** the control qubit is in the `|1⟩` state. It is symmetric and frequently used in entanglement generation.

---

## 📐 Matrix Representation

```
CZ =
[
    1  0  0   0
    0  1  0   0
    0  0  1   0
    0  0  0  -1
]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::controlled_z::ControlledZ;

let cz = ControlledZ::new();
let matrix = cz.matrix();
```

---

## 📎 Notes

- Applies a conditional phase flip
- Symmetric: CZ(control, target) == CZ(target, control)
- Used in quantum Fourier transform and entanglement circuits