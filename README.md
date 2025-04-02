# 🧠 QLang — Simulador Quântico em Rust com Backend CPU/GPU

Um comparativo estratégico com os principais simuladores quânticos do mercado.

| Recurso                       | QLang (Você) | Simuladores Top  |
| ----------------------------- | ------------ | ---------------- |
| Hadamard, Pauli, CNOT         | ✅ Sim       | ✅ Sim           |
| Medição                       | ✅ Sim       | ✅ Sim           |
| Rotações arbitrárias          | ✅ Sim       | ✅ Sim           |
| Portas de fase (S, T, etc)    | ✅ Sim       | ✅ Sim           |
| Ruído / Decoerência           | ❌ Não       | ✅ Sim           |
| Otimização de circuitos       | ❌ Não       | ✅ Sim           |
| Suporte a muitos qubits (>30) | ✅ Sim       | ✅ Sim (qsim...) |

# 🗺️ Mapa de Recursos — QLang vs Simuladores Quânticos

Comparação entre sua QVM + QLang e os principais simuladores (Qiskit, Cirq, qsim, etc).

| Recurso / Funcionalidade             | QLang (Você)   | Simuladores Top       | Próximo Passo Estratégico              |
| ------------------------------------ | -------------- | --------------------- | -------------------------------------- |
| **Gates básicos (Hadamard, Pauli)**  | ✅ Sim         | ✅ Sim                | ✔️ Já está no mesmo nível              |
| **CNOT**                             | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Medição (individual/all)**         | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Gates de rotação (RZ, RX, RY)**    | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Portas de fase (S, T)**            | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **SWAP**                             | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Toffoli (CCNOT)**                  | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
| **Fredkin (CSWAP)**                  | ✅ Sim         | ✅ Sim                | ✔️ Pronto                              |
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

- [x] Hadamard, Pauli (X, Y, Z)
- [x] CNOT
- [x] Medição
- [x] QLang + CLI
- [x] API via `lib.rs` (C/Python)

### 🔜 Versão 0.2 (meta atual):

| Ordem | Recurso / Gate   | Descrição                                   |
| ----- | ---------------- | ------------------------------------------- |
| [x]   | `rz(qubit, θ)`   | Rotação arbitrária no eixo Z (2x2 unitário) |
| [x]   | `s`, `t` gates   | Portas de fase (S = Rz(π/2), T = Rz(π/4))   |
| [x]   | `swap(q1, q2)`   | Implementado via 3 CNOTs                    |
| [x]   | `toffoli(a,b,c)` | Porta CCNOT (controle duplo) - 8x8 matriz   |
| [x]   | `fredkin(a,b,c)` | Porta CSWAP (troca controlada)              |

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

- [ ] **Paralelismo com `rayon` no backend CPU**  
       Use processamento multi-thread para acelerar multiplicações de matriz-vetor.  
       👉 Ideal para simular até 24–26 qubits com boa performance em CPUs modernas.

- [ ] **Suporte a circuito em batch (gate fusion)**  
       Otimize várias portas aplicadas em sequência no mesmo qubit.  
       👉 Reduz operações redundantes e melhora desempenho computacional.

- [ ] **Noise Modeling (modelo de ruído simples)**  
       Adicione ruído de depolarização, bit-flip, phase-flip etc.  
       👉 Essencial para simular circuitos realistas e avaliar tolerância a erros.

- [ ] **Exportação para QASM ou integração com Qiskit**  
       Gere `.qasm` ou permita importação/exportação direta para Qiskit.  
       👉 Permite rodar circuitos reais ou integrá-los a pipelines existentes.

---

## 🤝 Contribua ou acompanhe

Quer contribuir, sugerir uma feature ou usar QLang no seu projeto acadêmico ou empresarial?  
Sinta-se livre para abrir uma issue, mandar PR ou entrar em contato comigo.

---
