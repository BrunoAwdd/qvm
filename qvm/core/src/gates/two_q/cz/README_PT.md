# 🔒 Porta `Controlled-Z (CZ)` – Simulação Quântica QLang

A porta `CZ` é uma **porta Z controlada de dois qubits**. Ela aplica um Pauli-Z ao qubit alvo **somente quando** o qubit de controle está no estado `|1⟩`. É simétrica e amplamente utilizada para gerar entrelaçamento.

---

## 📐 Representação Matricial

```
CZ =
[
    1  0  0   0
    0  1  0   0
    0  0  1   0
    0  0  0  -1
]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::controlled_z::ControlledZ;

let cz = ControlledZ::new();
let matrix = cz.matrix();
```

---

## 📎 Notas

- Aplica uma inversão de fase condicional
- Simétrica: CZ(controle, alvo) == CZ(alvo, controle)
- Muito usada na transformada de Fourier quântica e em circuitos de entrelaçamento