# 🎯 `Controlled-Y (CY)` Gate – QLang Quantum Simulation

The `CY` gate is a **controlled Pauli-Y gate**. It applies the Y operation to the target qubit **only if** the control qubit is `|1⟩`.

---

## 📐 Matrix Representation

```
CY =
[
    1  0   0    0
    0  1   0    0
    0  0   0  -i
    0  0   i   0
]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::controlled_y::ControlledY;

let cy = ControlledY::new();
let matrix = cy.matrix();
```

---

## 📎 Notes

- Control qubit: 0
- Target qubit: 1
- Pauli-Y introduces both a bit and phase flip (`|0⟩ → i|1⟩`, `|1⟩ → -i|0⟩`)