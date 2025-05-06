# 🎯 `Toffoli` Gate – QLang Quantum Simulation

The `Toffoli` gate (also known as `CCX` or controlled-controlled-NOT) is a **three-qubit gate** that flips the target qubit **only if both control qubits are in the `|1⟩` state**. It is a key component in reversible and classical logic emulation on quantum circuits.

---

## 📐 Matrix Representation

The Toffoli gate is an 8×8 unitary matrix. It behaves as:

```
|a⟩|b⟩|c⟩ → |a⟩|b⟩|c ⊕ (a ∧ b)⟩
```

Only swaps states 6 and 7 (binary `110` and `111`).

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::toffoli::Toffoli;

let toffoli = Toffoli::new();
let matrix = toffoli.matrix();
```

---

## 📎 Notes

- Control qubits: 0 and 1 (most significant)
- Target qubit: 2 (least significant)
- Useful for quantum arithmetic and error correction logic