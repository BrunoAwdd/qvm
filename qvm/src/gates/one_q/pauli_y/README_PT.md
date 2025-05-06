# 🌀 Porta `Pauli-Y` – Simulação Quântica QLang

A porta `Pauli-Y` é uma porta quântica fundamental de **um único qubit**, que realiza uma inversão do qubit (como a Pauli-X), mas também aplica uma fase complexa (`±i`). É essencial em algoritmos quânticos que envolvem manipulação de fase.

---

## 📐 Representação Matricial

```
|  0  -i |
|  i   0 |
```

Quando aplicada aos estados base:

- `Y|0⟩ = i|1⟩`
- `Y|1⟩ = -i|0⟩`

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::pauli_y::PauliY;

let y = PauliY::new();
let matrix = y.matrix();
```

---

## 🚀 Kernel CUDA

```cpp
__global__ void pauli_y_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
);
```

- Aplica `Y` com multiplicação complexa manual por ±i para desempenho.
- Apenas threads com `i < partner` realizam a troca para evitar duplicidade.

---

## 🧪 Testes

- `test_pauli_y_matrix` – Valida que a matriz está correta conforme o esperado.
- `test_pauli_y_name` – Verifica se o nome da porta é `"pauliY"`.

---

## 📎 Notas

- Introduz inversão de bit e também de fase.
- Fundamental para operações quânticas com manipulação de fase complexa.