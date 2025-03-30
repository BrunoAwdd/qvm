# Criar gates

## Swap

## Toffoli

## Fredkin

Recurso Você tem? Top Simuladores
Hadamard, Pauli, CNOT Sim Sim
Medição ? Sim
Rotações arbitrárias Não Sim
Portas de fase (S, T, etc) Não Sim
Ruído / Decoerência Não Sim
Otimização de circuitos Não Sim
Suporte a muitos qubits (>30) Depende Sim (qsim, etc)

# 🗺️ Mapa de Recursos — QLang vs Simuladores Quânticos

Comparação entre sua QVM + QLang e os principais simuladores (Qiskit, Cirq, qsim, etc).

| Recurso / Funcionalidade             | QLang (Você)   | Simuladores Top       | Próximo Passo Estratégico              |
| ------------------------------------ | -------------- | --------------------- | -------------------------------------- |
| **Gates básicos (Hadamard, Pauli)**  | ✅ Sim         | ✅ Sim                | ✔️ Já está no mesmo nível              |
| **CNOT**                             | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Medição (individual/all)**         | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Gates de rotação (RZ, RX, RY)**    | ❌ Ainda não   | ✅ Sim                | 🔜 **Adicionar `rz` primeiro**         |
| **Portas de fase (S, T)**            | ❌ Ainda não   | ✅ Sim                | 🔜 Depois do `rz`, são simples         |
| **SWAP**                             | ❌ Ainda não   | ✅ Sim                | 🔜 Implementar com 3 CNOTs             |
| **Toffoli (CCNOT)**                  | ❌ Ainda não   | ✅ Sim                | 🔜 Matriz 8x8 com suporte de 3 qubits  |
| **Fredkin (CSWAP)**                  | ❌ Ainda não   | ✅ Sim                | 🔜 Implementar via matriz ou controle  |
| **Gate arbitrário unitário (U3)**    | ❌ Ainda não   | ✅ Sim                | ⚠️ Pode vir depois                     |
| **Ruído / Decoerência**              | ❌ Ainda não   | ✅ Sim                | 🚧 Avançado, deixar para versão futura |
| **Otimização de circuitos**          | ❌ Ainda não   | ✅ Sim                | 🧠 Fase 2 (análise e reordenação)      |
| **Suporte a >30 qubits**             | ⚠️ Parcial     | ✅ Sim (ex: qsim)     | 🔬 Testar escalabilidade da sua QVM    |
| **Execução via linguagem própria**   | ✅ Sim (QLang) | ⚠️ Alguns (QASM, etc) | ✔️ Você tem vantagem aqui 💎           |
| **Execução via string inline**       | ✅ Sim         | ✅ Sim                | ✔️ Com `run_qlang_inline`              |
| **API C / Python / FFI**             | ✅ Sim         | ✅ Sim                | ✔️ No mesmo nível                      |
| **REPL / Terminal interativo**       | ❌ Ainda não   | ⚠️ Poucos oferecem    | 🧪 Pode vir depois                     |
| **Visualização (circuitos/estados)** | ❌ Ainda não   | ✅ Sim                | 🖼️ Pode integrar com Python            |

---

## 🎯 Roadmap sugerido

### ✅ Já implementado:

- Hadamard, Pauli (X, Y, Z)
- CNOT
- Medição
- QLang + CLI
- API via `lib.rs` (C/Python)

### 🔜 Versão 0.2 (meta atual):

| Ordem | Recurso / Gate   | Descrição                                   |
| ----- | ---------------- | ------------------------------------------- |
| 1️⃣    | `rz(qubit, θ)`   | Rotação arbitrária no eixo Z (2x2 unitário) |
| 2️⃣    | `s`, `t` gates   | Portas de fase (S = Rz(π/2), T = Rz(π/4))   |
| 3️⃣    | `swap(q1, q2)`   | Implementado via 3 CNOTs                    |
| 4️⃣    | `toffoli(a,b,c)` | Porta CCNOT (controle duplo) - 8x8 matriz   |
| 5️⃣    | `fredkin(a,b,c)` | Porta CSWAP (troca controlada)              |

---

## ⚙️ Roadmap Paralelo — Suporte a GPU

### 🎯 Objetivo:

Acelerar simulações (produtos de matrizes, vetores de estado) com suporte real à GPU, sem ficar preso a apenas uma plataforma.

### 🧠 Avaliação das opções:

| Tecnologia GPU    | Prós                                          | Contras                            | Recomendado?          |
| ----------------- | --------------------------------------------- | ---------------------------------- | --------------------- |
| **cust (CUDA)**   | Máximo desempenho, fácil no Rust              | ❌ Só NVIDIA, drivers pesados      | ⚠️ Bom para benchmark |
| **opencl3**       | Roda em Intel, AMD, NVIDIA                    | API antiga, drivers inconsistentes | ⚠️ Alternativa neutra |
| **wgpu (WebGPU)** | 🚀 Cross-platform, moderna, suportada em WASM | Acesso mais indireto à GPU crua    | ✅ **Melhor escolha** |

---

### 🚀 Estratégia recomendada:

- ✅ Começar com `wgpu` e shaders computacionais (`WGSL`)
- ⚙️ Design modular com suporte a múltiplos backends
- 💡 Possível enum para alternar entre `CPU`, `GPU_CUDA`, `GPU_WGPU`

```rust
enum QVMBinding {
    CPU,
    GPU_CUDA,
    GPU_WGPU,
}
```
