# 🧠 `ControlledU` Gate – QLang Quantum Simulation

O `ControlledU` é um **gate de dois qubits** que aplica uma matriz unitária arbitrária ao qubit alvo **somente quando o qubit de controle estiver no estado `|1⟩`**. Ele é fundamental para a construção de portas como `CNOT`, `CZ`, ou mesmo `Controlled-Rx`, `Ry`, etc.

---

## 📐 Representação da Matriz

A matriz resultante do `ControlledU` é uma matriz 4×4 no seguinte formato:

```
| I  0 |
| 0  U |
```

- `I`: Identidade 2×2 → nada é aplicado se controle = `|0⟩`
- `U`: Matriz unitária 2×2 → aplicada se controle = `|1⟩`

---

## 🧰 Como usar (Rust)

```rust
use qlang::gates::controlled_u::ControlledU;

let cu = ControlledU::new_real(0.0, 1.0, 1.0, 0.0); // Simula um CNOT gate
let matrix = cu.matrix();
```

Ou com uma matriz complexa:

```rust
use ndarray::array;
use qlang::types::qlang_complex::QLangComplex;

let u = array![
    [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
    [QLangComplex::new(0.0, 0.0), QLangComplex::new(-1.0, 0.0)],
];
let cu = ControlledU::new(u, None);
```

---

## 🚀 CUDA Kernel

O projeto também inclui uma versão GPU do `ControlledU`, escrita em CUDA, para aplicação direta sobre o vetor de estado.

### Assinatura do Kernel

```cpp
__global__ void controlled_u_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits,
    cuDoubleComplex u00,
    cuDoubleComplex u01,
    cuDoubleComplex u10,
    cuDoubleComplex u11
);
```

Esse kernel aplica `U` ao qubit `target` **somente** quando o qubit `control` estiver em `|1⟩`. A implementação é segura contra race conditions por design.

---

## 🧪 Testes

- `test_controlled_u_from_real` – Verifica o comportamento com valores reais (ex: CNOT).
- `test_controlled_u_from_matrix` – Garante que a matriz `U` é corretamente posicionada na matriz 4×4.
- `test_controlled_u_name` – Confirma a identificação textual do gate como `"cu"`.

---

## 📎 Notas

- Exige que a matriz `U` fornecida seja unitária (2×2).
- A versão CUDA assume `target != control`.
- Para sistemas maiores, a performance do kernel é crítica e foi projetada com eficiência em mente.
