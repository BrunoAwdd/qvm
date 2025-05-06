# 🔺 Porta `Phase(θ)` – Simulação Quântica QLang

A porta `Phase` é uma **porta de fase de um único qubit generalizada**, que aplica uma fase complexa `θ` (em radianos) ao estado `|1⟩`. Comportamentos das portas S, T e T† são casos particulares.

---

## 📐 Representação Matricial

```
| 1       0 |
| 0   e^{iθ} |
```

Comportamento:

- `Phase(θ)|0⟩ = |0⟩`
- `Phase(θ)|1⟩ = e^{iθ}|1⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::phase::Phase;

let theta = std::f64::consts::PI / 2.0;
let p = Phase::new(theta);
let matrix = p.matrix();
```

---

## ⛓️ Casos Especiais

- `Phase(π/2)` → porta `S`
- `Phase(π/4)` → porta `T`
- `Phase(−π/4)` → porta `T†`

---

## 🧪 Testes

- `test_phase_gate_matrix_theta_pi_2` – Valida a matriz para `θ = π/2`.
- `test_phase_gate_name` – Verifica que o nome da porta é `"phase"`.

---

## 📎 Notas

- A fase é aplicada somente ao estado `|1⟩`.
- Muito usada em algoritmos que exigem controle de fase variável.