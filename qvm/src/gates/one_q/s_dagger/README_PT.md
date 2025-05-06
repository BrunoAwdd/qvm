# 🔷 Porta `S†` – Simulação Quântica QLang

A porta `S†` (S-dagger) é a **inversa** da porta de fase `S`. Ela aplica uma fase negativa de π/2 (−i) ao estado `|1⟩` e é frequentemente usada em reversões e correções de circuitos quânticos.

---

## 📐 Representação Matricial

```
| 1    0 |
| 0  -i |
```

Comportamento:

- `S†|0⟩ = |0⟩`
- `S†|1⟩ = -i|1⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::s_dagger::SDagger;

let s_dag = SDagger::new();
let matrix = s_dag.matrix();
```

---

## 🧪 Testes

- `test_s_dagger_matrix` – Valida a implementação da matriz.
- `test_s_dagger_name` – Verifica se o nome da porta é `"SDagger"`.

---

## 📎 Notas

- `S† * S = Identidade`
- Usada para reverter operações quânticas e aplicar correção de fase.