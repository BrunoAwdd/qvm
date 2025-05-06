# 🟣 Porta `T` – Simulação Quântica QLang

A porta `T` (π/8) é uma **porta de fase de um único qubit** que aplica uma fase de π/4 ao estado `|1⟩`. É uma porta não-Clifford e essencial para a computação quântica universal.

---

## 📐 Representação Matricial

```
| 1         0 |
| 0  e^(iπ/4) |
```

Comportamento:

- `T|0⟩ = |0⟩`
- `T|1⟩ = e^(iπ/4)|1⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::t::T;

let t = T::new();
let matrix = t.matrix();
```

---

## 🧪 Testes (sugerido)

- Validar se a matriz corresponde à fase esperada.
- Confirmar se o nome da porta é `"T"`.

---

## 📎 Notas

- `T * T = S`
- Inversa: `T†` (fase −π/4)
- Garante universalidade quando combinada com portas Clifford (H, S, CNOT).