# 🔷 `S` Gate – QLang Quantum Simulation

The `S` gate is a **single-qubit phase gate** that applies a π/2 phase (i) to the `|1⟩` state. It is also referred to as the **√Z** gate since it is the square root of the Pauli-Z gate.

---

## 📐 Matrix Representation

```
| 1   0 |
| 0   i |
```

Its behavior:

- `S|0⟩ = |0⟩`
- `S|1⟩ = i|1⟩`

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::s::S;

let s = S::new();
let matrix = s.matrix();
```

---

## 🧪 Tests (suggested)

- Validate that the matrix matches the theoretical S gate.
- Check that the gate is named `"S"`.

---

## 📎 Notes

- It is a Clifford gate and useful in quantum error correction.
- Applying `S` twice is equivalent to applying the Pauli-Z gate (`S² = Z`).