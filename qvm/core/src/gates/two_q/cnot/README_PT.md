# ⚡ Porta `CNOT` – Simulação Quântica QLang

A porta `CNOT` (NOT controlado) é uma **porta de dois qubits** que inverte o qubit alvo **somente se** o qubit de controle estiver no estado `|1⟩`. É uma das portas mais fundamentais da computação quântica, permitindo entrelaçamento e lógica condicional.

---

## 📐 Representação Matricial

```
CNOT =
[
    1  0  0  0
    0  1  0  0
    0  0  0  1
    0  0  1  0
]
```

Essa matriz troca os estados `|10⟩` e `|11⟩`.

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::cnot::CNOT;

let cnot = CNOT::new();
let matrix = cnot.matrix();
```

---

## 📎 Notas

- Controle: qubit 0
- Alvo: qubit 1
- Essencial para entrelaçamento e lógica universal