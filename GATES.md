# Portas Suportadas no Qlang/QVM

Este documento lista todas as portas quânticas suportadas (ou planejadas) na linguagem Qlang e no simulador QVM, com explicações rápidas e status de implementação.

---

## 1. Portas de 1 Qubit

| Nome       | Código | Função                         | Status          |
| ---------- | ------ | ------------------------------ | --------------- |
| Pauli-X    | x(q)   | Inverte o estado: 0⟩ ↔ 1⟩      | ✅ Implementado |
| Pauli-Y    | y(q)   | Rotação com fase imaginária    | ✅ Implementado |
| Pauli-Z    | z(q)   | Aplica fase -1 ao 1⟩           | ✅ Implementado |
| Hadamard   | h(q)   | Cria superposição: (0⟩ +1⟩)/√2 | ✅ Implementado |
| S          | s(q)   | Aplica fase de π/2             | ✅ Implementado |
| S-dagger   | sdg(q) | Inversa da S                   | ✅ Implementado |
| T          | t(q)   | Aplica fase de π/4             | ✅ Implementado |
| T-dagger   | tdg(q) | Inversa da T                   | ✅ Implementado |
| Identidade | id(q)  | Não faz nada                   | ✅ Implementado |

---

## 2. Portas de Rotação

| Nome  | Código         | Função                     | Status          |
| ----- | -------------- | -------------------------- | --------------- |
| Rx    | rx(θ, q)       | Rotação no eixo X          | ✅ Implementado |
| Ry    | ry(θ, q)       | Rotação no eixo Y          | ✅ Implementado |
| Rz    | rz(θ, q)       | Rotação no eixo Z          | ✅ Implementado |
| U3    | u3(θ, φ, λ, q) | Porta universal 1-qubit    | ✅ Implementado |
| U2    | u2(φ, λ, q)    | Intermediário entre H e U3 | ⬜ Planejado    |
| U1    | u1(λ, q)       | Rz com fator de escala     | ⬜ Planejado    |
| Phase | phase(θ, q)    | Aplica fase arbitrária     | ⬜ Planejado    |

---

## 3. Portas de 2 Qubits (Controladas)

| Nome  | Código      | Função                               | Status          |
| ----- | ----------- | ------------------------------------ | --------------- |
| CNOT  | cnot(c, t)  | Controla X no alvo se controle for 1 | ✅ Implementado |
| CZ    | cz(c, t)    | Controla Z no alvo se controle for 1 | ⬜ Planejado    |
| CY    | cy(c, t)    | Controla Y no alvo se controle for 1 | ⬜ Planejado    |
| SWAP  | swap(a, b)  | Troca os estados de dois qubits      | ✅ Implementado |
| iSWAP | iswap(a, b) | Troca com fase imaginária            | ⬜ Planejado    |

---

## 4. Portas de 3 Qubits

| Nome    | Código             | Função                                 | Status          |
| ------- | ------------------ | -------------------------------------- | --------------- |
| Toffoli | toffoli(c1, c2, t) | Aplica X se ambos os controles forem 1 | ✅ Implementado |
| Fredkin | fredkin(c, q1, q2) | Troca dois alvos se controle for 1     | ✅ Implementado |

---

## 5. Medidas

| Nome     | Código    | Função                         | Status                           |
| -------- | --------- | ------------------------------ | -------------------------------- |
| Medida Z | m(q) / mz | Mede na base computacional (Z) | ✅ Implementado                  |
| Medida X | mx(q)     | Aplica H e mede (base X)       | ⬜ Planejado (via Python helper) |
| Medida Y | my(q)     | Aplica S† + H e mede (base Y)  | ⬜ Planejado (via Python helper) |

---

## Legenda

- ✅ _Implementado_: Disponível na linguagem e no simulador
- ⬜ _Planejado_: Recurso previsto ou em desenvolvimento

---

## Contribuições

Se quiser contribuir com novas portas, otimizações ou melhorias, veja o arquivo CONTRIBUTING.md ou abra uma issue no repositório.
