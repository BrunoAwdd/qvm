# 🧠 `U3(θ, φ, λ)` Gate – QLang Quantum Simulation

The `U3` gate is the most general single-qubit gate, capable of representing any unitary operation on one qubit. It is parameterized by three angles: θ (theta), φ (phi), and λ (lambda), and is a standard in OpenQASM and Qiskit.

---

## 📐 Matrix Representation

```
U3(θ, φ, λ) =
[
    cos(θ/2)             -e^{iλ}·sin(θ/2)
    e^{iφ}·sin(θ/2)      e^{i(φ+λ)}·cos(θ/2)
]
```

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::u3::U3;

let theta = std::f64::consts::PI / 2.0;
let phi = std::f64::consts::PI / 4.0;
let lambda = std::f64::consts::PI / 3.0;

let u3 = U3::new(theta, phi, lambda);
let matrix = u3.matrix();
```

---

## 📎 Notes

- U3 is a superset of U1 and U2 gates:
  - `U3(0, 0, λ) = U1(λ)`
  - `U3(π/2, φ, λ) = U2(φ, λ)`
- This gate is universal for single-qubit operations.