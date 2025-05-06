# 🔷 Porta `S` – Simulação Quântica QLang

A porta `S` é uma **porta de fase de um único qubit** que aplica uma fase de π/2 (i) ao estado `|1⟩`. Também é chamada de porta **√Z**, pois é a raiz quadrada da porta Pauli-Z.

---

## 📐 Representação Matricial

```
| 1   0 |
| 0   i |
```

Comportamento:

- `S|0⟩ = |0⟩`
- `S|1⟩ = i|1⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::s::S;

let s = S::new();
let matrix = s.matrix();
```

---

## 🧪 Testes (sugerido)

- Verificar se a matriz corresponde à teoria da porta S.
- Confirmar se o nome da porta é `"S"`.

---

## 📎 Notas

- É uma porta de Clifford, útil em correção de erros quânticos.
- Aplicar `S` duas vezes equivale a aplicar a porta Pauli-Z (`S² = Z`).