# 🟣 Porta `T†` – Simulação Quântica QLang

A porta `T†` (T-dagger) é a **inversa** da porta `T` (π/8). Ela aplica uma fase de −π/4 ao estado `|1⟩` e é fundamental para reversões e correções de erros em circuitos quânticos.

---

## 📐 Representação Matricial

```
| 1           0 |
| 0   e^(-iπ/4) |
```

Comportamento:

- `T†|0⟩ = |0⟩`
- `T†|1⟩ = e^(-iπ/4)|1⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::t_dagger::TDagger;

let t_dag = TDagger::new();
let matrix = t_dag.matrix();
```

---

## 🧪 Testes

- `test_t_dagger_matrix` – Verifica a matriz com fase −π/4.
- `test_t_dagger_name` – Confirma que o nome da porta é `"TDagger"`.

---

## 📎 Notas

- Conjugado Hermitiano da porta `T`.
- `T† * T = Identidade`
- Importante em circuitos reversíveis e com tolerância a falhas.