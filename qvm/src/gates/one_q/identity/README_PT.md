# 🟦 Porta `Identidade` – Simulação Quântica QLang

A porta `Identidade` (I) é uma **porta de não-operação (no-op)** que não altera o estado do qubit. Ela representa a matriz identidade na computação quântica.

---

## 📐 Representação Matricial

```
| 1  0 |
| 0  1 |
```

Essa porta não modifica a amplitude do qubit. É utilizada em:
- Sincronização entre qubits
- Operações de preenchimento (placeholder)
- Testes ou portas compostas

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::identity::Identity;

let id = Identity::new();
let matrix = id.matrix();
```

---

## 🧪 Testes

- `test_identity_matrix_correctness` – Confirma a estrutura da matriz identidade.
- `test_identity_name` – Verifica o nome da porta como `"Identity"`.

---

## 📎 Notas

- Não afeta a evolução do estado quântico.
- Útil para testes, controle de fluxo ou alinhamento de tempos de portas.