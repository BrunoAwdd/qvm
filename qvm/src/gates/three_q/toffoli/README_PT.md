# 🎯 Porta `Toffoli` – Simulação Quântica QLang

A porta `Toffoli` (também conhecida como `CCX` ou NOT controlado por dois qubits) é uma **porta de três qubits** que inverte o qubit alvo **somente se os dois qubits de controle estiverem no estado `|1⟩`**. É essencial em computação reversível e emulação de lógica clássica.

---

## 📐 Representação Matricial

A porta Toffoli é uma matriz unitária 8×8. Seu comportamento é:

```
|a⟩|b⟩|c⟩ → |a⟩|b⟩|c ⊕ (a ∧ b)⟩
```

Somente as posições 6 e 7 (binário `110` e `111`) são trocadas.

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::toffoli::Toffoli;

let toffoli = Toffoli::new();
let matrix = toffoli.matrix();
```

---

## 📎 Notas

- Qubits de controle: 0 e 1 (mais significativos)
- Qubit alvo: 2 (menos significativo)
- Muito usada em aritmética quântica e lógica de correção de erros