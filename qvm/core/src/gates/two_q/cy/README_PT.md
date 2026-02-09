# 🎯 Porta `Controlled-Y (CY)` – Simulação Quântica QLang

A porta `CY` é uma **porta Pauli-Y controlada**. Ela aplica a operação Y ao qubit alvo **somente se** o qubit de controle estiver em `|1⟩`.

---

## 📐 Representação Matricial

```
CY =
[
    1  0   0    0
    0  1   0    0
    0  0   0  -i
    0  0   i   0
]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::controlled_y::ControlledY;

let cy = ControlledY::new();
let matrix = cy.matrix();
```

---

## 📎 Notas

- Qubit de controle: 0
- Qubit alvo: 1
- Pauli-Y combina inversão de bit com mudança de fase (`|0⟩ → i|1⟩`, `|1⟩ → -i|0⟩`)