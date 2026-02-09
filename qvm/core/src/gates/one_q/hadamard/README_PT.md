# ✨ Porta `Hadamard` – Simulação Quântica QLang

A porta `Hadamard` (H) é uma porta quântica fundamental de um único qubit que cria **superposição**. Quando aplicada a um qubit, ela transforma os estados base em superposições iguais:

- `H|0⟩ = (|0⟩ + |1⟩)/√2`
- `H|1⟩ = (|0⟩ - |1⟩)/√2`

---

## 📐 Representação da Matriz

A matriz da porta Hadamard é:

```
1/√2 * |  1   1 |
        |  1  -1 |
```

Ela é **hermitiana** e **unitária**, ou seja, sua própria inversa.

---

## 🧰 Uso (em Rust)

```rust
use qlang::gates::one_q::hadamard::Hadamard;

let h = Hadamard::new();
let matrix = h.matrix();
```

Aplicando em uma `QVM`:

```rust
let mut qvm = QVM::new(1);
qvm.apply_gate(&Hadamard::new(), 0);
```

---

## 🚀 Kernel CUDA

A porta Hadamard também pode ser aplicada de forma paralela no vetor de estado quântico via CUDA:

```cpp
__global__ void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits);
```

- Aplica H de forma eficiente ao `qubit` especificado no vetor `state`
- Somente uma thread por par aplica a atualização, evitando condições de corrida
- Utiliza precisão dupla (`cuDoubleComplex`) para representar amplitudes

---

## 🧪 Testes

- `test_hadamard_matrix` – Confirma se a matriz está correta.
- `test_hadamard_name` – Verifica a identificação textual da porta.
- `test_hadamard_apply` – Aplica H e observa comportamento probabilístico.
- `test_hadamard_distribution` – Garante distribuição 50/50 em múltiplas execuções.
- `test_measure_many_hadamard` – Teste de integração com comandos `QLang` e múltiplas medições.

---

## 📎 Notas

- Componente essencial em algoritmos quânticos como Grover e Shor.
- Utilizada para inicializar qubits em superposição antes de computações baseadas em interferência.