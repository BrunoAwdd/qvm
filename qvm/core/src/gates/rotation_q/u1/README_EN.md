# 🧭 `U1(λ)` Gate – QLang Quantum Simulation

The `U1` gate is a **single-parameter phase gate** that applies a phase shift `λ` (lambda) to the `|1⟩` state, leaving `|0⟩` unchanged. It is equivalent to a `RZ(λ)` rotation up to global phase.

---

## 📐 Matrix Representation

```
U1(λ) =
[ 1       0     ]
[ 0   e^{iλ}    ]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::u1::U1;

let lambda = std::f64::consts::PI;
let u1 = U1::new(lambda);
let matrix = u1.matrix();
```

---

## 🧪 Tests (suggested)

- Validate that `U1(λ)` equals `Phase(λ)` or `RZ(λ)` up to global phase.
- Ensure the name is `"U1"`.

---

## 📎 Notes

- `U1` is part of the standard U-gate set in OpenQASM.
- It is the same as `Phase(λ)` in many implementations.