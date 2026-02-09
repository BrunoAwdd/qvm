# ⚡ `Pauli-Z` Gate – QLang Quantum Simulation

The `Pauli-Z` gate is a fundamental **single-qubit quantum gate** that performs a **phase flip** without changing the amplitude of the basis states.

---

## 📐 Matrix Representation

```
| 1   0 |
| 0  -1 |
```

Its behavior:

- `Z|0⟩ = |0⟩`
- `Z|1⟩ = -|1⟩`

This gate is often used in phase-based quantum algorithms and plays a crucial role in interference.

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::pauli_z::PauliZ;

let z = PauliZ::new();
let matrix = z.matrix();
```

---

## 🧪 Tests (suggested)

- Validate that the matrix matches the expected structure.
- Check the gate name is `"PauliZ"`.

---

## 📎 Notes

- Common in algorithms involving phase kickback or phase estimation.
- Used to construct the S and T gates (via decompositions).