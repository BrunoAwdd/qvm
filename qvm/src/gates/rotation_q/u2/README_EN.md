# 🧮 `U2(φ, λ)` Gate – QLang Quantum Simulation

The `U2` gate is a **parametric single-qubit gate** that performs a unitary rotation with phase shifts `φ` and `λ`. It is a commonly used gate in Qiskit and other quantum programming frameworks for implementing general rotations.

---

## 📐 Matrix Representation

```
U2(φ, λ) = (1/√2) ×
[  1           -e^{iλ}       ]
[  e^{iφ}   e^{i(φ+λ)}       ]
```

Equivalent to a `U3(π/2, φ, λ)` gate.

---

## 🧰 Usage (Rust)

```rust
use qlang::gates::u2::U2;

let phi = std::f64::consts::FRAC_PI_2;
let lambda = std::f64::consts::FRAC_PI_4;
let u2 = U2::new(phi, lambda);
let matrix = u2.matrix();
```

---

## 🧪 Tests

- `test_u2_matrix` – Validates matrix generation for known φ, λ.
- `test_u2_name` – Confirms that the gate is named `"U2"`.

---

## 📎 Notes

- U2 is a core gate in OpenQASM standard circuits.
- U2(0, π) is equivalent to the Hadamard gate.