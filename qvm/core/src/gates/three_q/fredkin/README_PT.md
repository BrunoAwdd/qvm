# 🔁 Porta `Fredkin` – Simulação Quântica QLang

A porta `Fredkin` é uma **porta SWAP controlada**. Ela troca dois qubits alvo **somente se** o qubit de controle estiver no estado `|1⟩`. É uma porta de 3 qubits com aplicações em computação reversível e lógica quântica.

---

## 📐 Representação Matricial

A porta Fredkin é uma matriz 8×8. Seu efeito é:

```
|c⟩|a⟩|b⟩ → se c = 1: troca(a, b), senão: nada muda
```

Apenas a submatriz para controle = 1 e a ≠ b é afetada (posições 5 e 6 na indexação binária).

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::fredkin::Fredkin;

let fredkin = Fredkin::new();
let matrix = fredkin.matrix();
```

---

## 📎 Notas

- Qubit de controle: qubit 0 (mais significativo)
- Alvos: qubit 1 e qubit 2
- Usada em multiplexadores quânticos e correção de erros