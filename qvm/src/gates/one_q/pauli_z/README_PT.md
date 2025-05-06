# ⚡ Porta `Pauli-Z` – Simulação Quântica QLang

A porta `Pauli-Z` é uma porta quântica fundamental de **um único qubit** que realiza uma **inversão de fase**, sem alterar a amplitude dos estados base.

---

## 📐 Representação Matricial

```
| 1   0 |
| 0  -1 |
```

Comportamento:

- `Z|0⟩ = |0⟩`
- `Z|1⟩ = -|1⟩`

Essa porta é amplamente usada em algoritmos baseados em fase e desempenha papel essencial em interferência quântica.

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::pauli_z::PauliZ;

let z = PauliZ::new();
let matrix = z.matrix();
```

---

## 🧪 Testes (sugerido)

- Verificar se a matriz está de acordo com a estrutura esperada.
- Confirmar que o nome da porta é `"PauliZ"`.

---

## 📎 Notas

- Comum em algoritmos com inversão de fase e estimativa de fase.
- Usada como base para a construção das portas S e T (via decomposição).