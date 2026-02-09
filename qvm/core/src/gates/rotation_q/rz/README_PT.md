# 🌀 Porta `RZ(θ)` – Simulação Quântica QLang

A porta `RZ` realiza uma **rotação em torno do eixo Z** da esfera de Bloch por um ângulo `θ` (em radianos). Ela aplica fases complexas opostas aos componentes `|0⟩` e `|1⟩`.

---

## 📐 Representação Matricial

```
RZ(θ) =
[ e^{-iθ/2}      0      ]
[     0      e^{iθ/2}   ]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::rz::RZ;

let theta = std::f64::consts::PI / 2.0;
let rz = RZ::new(theta);
let matrix = rz.matrix();
```

---

## 🧪 Testes (sugerido)

- Verificar se a matriz aplica as fases corretas ±θ/2.
- Confirmar que o nome da porta é `"RZ"`.

---

## 📎 Notas

- A `RZ` é diagonal e baseada unicamente em fase.
- Muito usada junto com `RX` e `RY` para criar rotações arbitrárias.
- Quando `θ = π`, a `RZ` age como uma Pauli-Z até uma fase global.