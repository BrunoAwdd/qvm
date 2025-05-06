# 🔁 Porta `iSWAP` – Simulação Quântica QLang

A porta `iSWAP` é uma porta quântica de dois qubits que **troca os estados `|01⟩` e `|10⟩`** com uma **fase imaginária `i`**. É útil em simulações, entrelaçamento e modelos inspirados em física.

---

## 📐 Representação Matricial

```
iSWAP =
[
    1   0   0   0
    0   0   i   0
    0   i   0   0
    0   0   0   1
]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::iswap::ISwap;

let gate = ISwap::new();
let matrix = gate.matrix();
```

---

## 📎 Notas

- Troca `|01⟩` ↔ `|10⟩` com uma mudança de fase `i`
- Usada para simular interações em modelos físicos