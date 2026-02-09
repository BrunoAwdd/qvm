# ⚡ `CNOT` Gate – QLang Quantum Simulation

The `CNOT` gate (controlled-NOT) is a **two-qubit gate** that flips the target qubit **only if** the control qubit is in the `|1⟩` state. It is one of the most fundamental gates in quantum computing, enabling entanglement and conditional logic.

---

## 📐 Matrix Representation

```
CNOT =
[
    1  0  0  0
    0  1  0  0
    0  0  0  1
    0  0  1  0
]
```

This gate swaps the states `|10⟩` and `|11⟩`.

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::cnot::CNOT;

let cnot = CNOT::new();
let matrix = cnot.matrix();
```

---

## 📎 Notes

- Control: qubit 0
- Target: qubit 1
- Crucial for entanglement and universal quantum logic