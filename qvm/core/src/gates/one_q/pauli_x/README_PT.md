# ❌ Porta `Pauli-X` – Simulação Quântica QLang

A porta `Pauli-X` (também chamada de porta `X` ou NOT quântica) inverte o estado de um qubit. É uma das portas fundamentais de Pauli utilizadas em computação quântica.

---

## 📐 Representação Matricial

```
| 0  1 |
| 1  0 |
```

Essa porta troca as amplitudes entre |0⟩ e |1⟩:

- `X|0⟩ = |1⟩`
- `X|1⟩ = |0⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::pauli_x::PauliX;

let x = PauliX::new();
let matrix = x.matrix();
```

---

## 🧪 Testes

- `test_pauli_x_matrix` – Valida a estrutura da matriz da porta Pauli-X.
- `test_pauli_x_name` – Confirma que o nome da porta é `"PauliX"`.

---

## 📎 Notas

- Equivalente ao NOT clássico no comportamento.
- Muito usada para inversão de bits e construção de operações mais complexas como o CNOT.