# 🔁 Porta `SWAP` – Simulação Quântica QLang

A porta `SWAP` é uma porta de dois qubits que **troca os estados dos dois qubits**.

---

## 📐 Representação Matricial

```
SWAP =
[
    1  0  0  0
    0  0  1  0
    0  1  0  0
    0  0  0  1
]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::swap::Swap;

let gate = Swap::new();
let matrix = gate.matrix();
```

---

## 📎 Notas

- Troca os estados `|01⟩` e `|10⟩`
- Muito utilizada para contornar limitações físicas de conectividade entre qubits