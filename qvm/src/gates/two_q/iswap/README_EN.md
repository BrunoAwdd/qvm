# 🔁 `iSWAP` Gate – QLang Quantum Simulation

The `iSWAP` gate is a two-qubit quantum gate that **swaps the `|01⟩` and `|10⟩` states** with an **imaginary phase `i`**. It is useful in simulation, entanglement, and condensed matter models.

---

## 📐 Matrix Representation

```
iSWAP =
[
    1   0   0   0
    0   0   i   0
    0   i   0   0
    0   0   0   1
]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::iswap::ISwap;

let gate = ISwap::new();
let matrix = gate.matrix();
```

---

## 📎 Notes

- Swaps `|01⟩` ↔ `|10⟩` with a phase shift of `i`
- Useful for simulating interactions in physics-inspired models