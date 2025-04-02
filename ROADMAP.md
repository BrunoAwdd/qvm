# Roadmap de Extensão da QLang QVM

Este roadmap foca em otimizações e expansões para o simulador quântico QLang, incluindo suporte a batching, slicing e tensor networks. O objetivo é escalar simulações com eficiência mesmo em máquinas modestas.

---

## ✅ Etapa Atual: Núcleo da QVM

- [x] Portas básicas implementadas (Hadamard, Pauli, R, CNOT, Toffoli, Fredkin)
- [x] Backend em Rust com suporte CPU e CUDA via cust
- [x] Execução via bindings C para qualquer linguagem (Python, etc.)
- [x] Full state vector simulator

---

## 1. Batching de Circuitos Pequenos (CUDA e CPU)

### Objetivo:

Rodar múltiplos circuitos pequenos em paralelo, melhorando o aproveitamento do CUDA (ou CPU multinúcleo).

### Tarefas:

- [ ] Abstrair estrutura de "job" (um circuito + estado inicial)
- [ ] Criar mecanismo de batch que recebe Vec<Circuit>
- [ ] Escrever kernel CUDA que opera em múltiplos vetores de estado simultaneamente
- [ ] Benchmark: batch vs single circuit

### Desafios:

- Gerenciar memória de vários vetores de estado simultaneamente
- Evitar divergência de execução entre os jobs (GPU gosta de uniformidade)

---

## 2. Slicing/Chunking do Vetor de Estado (CPU)

### Objetivo:

Permitir simulação de circuitos com mais de 32 qubits mesmo em PCs com pouca RAM.

### Tarefas:

- [ ] Implementar estrutura de StateChunk
- [ ] Carregar/aplicar operações em chunks de estado de forma incremental
- [ ] Usar disco como armazenamento intermediário para chunks fora da RAM
- [ ] Benchmark: impacto do disco em simulações grandes

### Desafios:

- Precisão numérica e sincronização entre chunks
- I/O de disco pode ser lento — otimizar com mmap ou caching

---

## 3. Tensor Networks (CPU)

### Objetivo:

Permitir simulação de circuitos maiores com menor custo de RAM, explorando baixa entanglement.

### Tarefas:

- [ ] Estudar representação de tensor networks (ex: MPS)
- [ ] Criar estrutura de TensorNode e TensorNetwork
- [ ] Implementar aplicação de portas como transformações de tensores
- [ ] Criar convertedor: circuito QLang → tensor network

### Desafios:

- Curva de aprendizado e matemática complexa
- Algoritmos de contração eficientes são não-triviais

### Referências:

- [Quimb](https://quimb.readthedocs.io/en/latest/)
- [ITensor](https://itensor.org/)
- [Google TN Sim](https://arxiv.org/abs/1808.00128)

---

## Futuro (Extras Possíveis)

- [ ] Estabilizadores (para circuitos só com Clifford)
- [ ] Simulação com ruído/qubits físicos
- [ ] Interface web com visualização de circuito e estado
- [ ] Integração com linguagens como Julia, Go, etc.

---

## Licença e Repositório

- Repositório: [Em breve...]
- Licença sugerida: MIT ou Apache 2.0
- Objetivo: ferramenta aberta, didática e poderosa

---
