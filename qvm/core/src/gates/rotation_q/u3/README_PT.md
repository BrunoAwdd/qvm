# 🧠 Porta `U3(θ, φ, λ)` – Simulação Quântica QLang

A porta `U3` é a forma mais geral de porta de 1 qubit, capaz de representar qualquer operação unitária. É parametrizada por três ângulos: θ (teta), φ (phi) e λ (lambda), sendo padrão no OpenQASM e Qiskit.

---

## 📐 Representação Matricial

```
U3(θ, φ, λ) =
[
    cos(θ/2)             -e^{iλ}·sin(θ/2)
    e^{iφ}·sin(θ/2)      e^{i(φ+λ)}·cos(θ/2)
]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::u3::U3;

let theta = std::f64::consts::PI / 2.0;
let phi = std::f64::consts::PI / 4.0;
let lambda = std::f64::consts::PI / 3.0;

let u3 = U3::new(theta, phi, lambda);
let matrix = u3.matrix();
```

---

## 📎 Notas

- A `U3` generaliza as portas `U1` e `U2`:
  - `U3(0, 0, λ) = U1(λ)`
  - `U3(π/2, φ, λ) = U2(φ, λ)`
- É universal para operações de um único qubit.