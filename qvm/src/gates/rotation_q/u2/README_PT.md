# 🧮 Porta `U2(φ, λ)` – Simulação Quântica QLang

A porta `U2` é uma **porta paramétrica de um único qubit** que realiza uma rotação unitária com fases `φ` e `λ`. É amplamente utilizada no Qiskit e em outros frameworks quânticos para decompor operações arbitrárias.

---

## 📐 Representação Matricial

```
U2(φ, λ) = (1/√2) ×
[  1           -e^{iλ}       ]
[  e^{iφ}   e^{i(φ+λ)}       ]
```

Equivalente a uma porta `U3(π/2, φ, λ)`.

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::u2::U2;

let phi = std::f64::consts::FRAC_PI_2;
let lambda = std::f64::consts::FRAC_PI_4;
let u2 = U2::new(phi, lambda);
let matrix = u2.matrix();
```

---

## 🧪 Testes

- `test_u2_matrix` – Valida a matriz para φ e λ conhecidos.
- `test_u2_name` – Confirma que o nome da porta é `"U2"`.

---

## 📎 Notas

- A U2 é uma porta fundamental nos circuitos OpenQASM.
- U2(0, π) é equivalente à porta de Hadamard.