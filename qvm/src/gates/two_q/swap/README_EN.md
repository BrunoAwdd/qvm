# 🔁 `SWAP` Gate – QLang Quantum Simulation

The `SWAP` gate is a two-qubit gate that **exchanges the states of the two qubits**.

---

## 📐 Matrix Representation

```
SWAP =
[
    1  0  0  0
    0  0  1  0
    0  1  0  0
    0  0  0  1
]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::swap::Swap;

let gate = Swap::new();
let matrix = gate.matrix();
```

---

## 📎 Notes

- Swaps the `|01⟩` and `|10⟩` states
- Frequently used in qubit routing and hardware constraint mitigation