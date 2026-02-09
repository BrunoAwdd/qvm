# 🔁 `Fredkin` Gate – QLang Quantum Simulation

The `Fredkin` gate is a **controlled-SWAP** gate. It swaps two target qubits **only if** the control qubit is in the `|1⟩` state. It is a 3-qubit gate with applications in reversible computing and quantum logic.

---

## 📐 Matrix Representation

The Fredkin gate is an 8×8 matrix. Its effect is:

```
|c⟩|a⟩|b⟩ → if c = 1: swap(a, b), else: no change
```

Only the submatrix for control = 1 and a ≠ b is affected (positions 5 and 6 in binary indexing).

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::fredkin::Fredkin;

let fredkin = Fredkin::new();
let matrix = fredkin.matrix();
```

---

## 📎 Notes

- Control qubit: qubit 0 (most significant bit)
- Targets: qubit 1 and qubit 2
- Often used in quantum multiplexers and error correction circuits