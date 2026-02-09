# 🧭 Índice das Portas Quânticas – QLang (PT)

Um índice completo das portas quânticas implementadas no simulador QLang, com links para cada documentação individual.

---

## 🔹 Portas de 1 Qubit

| Porta     | Descrição                           | Link |
|-----------|-------------------------------------|------|
| Identity  | Porta nula (sem operação)           | [README](README_Identity_PT.md) |
| Hadamard  | Cria superposição                   | [README](README_Hadamard_PT.md) |
| Pauli-X   | Inversão de bit (X)                 | [README](README_PauliX_PT.md) |
| Pauli-Y   | Inversão com fase                   | [README](README_PauliY_PT.md) |
| Pauli-Z   | Inversão de fase                    | [README](README_PauliZ_PT.md) |
| S         | Deslocamento de fase π/2            | [README](README_S_PT.md) |
| S†        | Inversa de S                        | [README](README_SDagger_PT.md) |
| T         | Deslocamento de fase π/4            | [README](README_T_PT.md) |
| T†        | Inversa de T                        | [README](README_TDagger_PT.md) |
| Phase     | Deslocamento de fase arbitrário     | [README](README_Phase_PT.md) |
| RX        | Rotação em torno do eixo X          | [README](README_RX_PT.md) |
| RY        | Rotação em torno do eixo Y          | [README](README_RY_PT.md) |
| RZ        | Rotação em torno do eixo Z          | [README](README_RZ_PT.md) |
| U1        | Rotação apenas de fase              | [README](README_U1_PT.md) |
| U2        | Porta paramétrica de 1 qubit        | [README](README_U2_PT.md) |
| U3        | Rotação geral de 1 qubit            | [README](README_U3_PT.md) |

---

## 🔹 Portas de 2 Qubits

| Porta     | Descrição                           | Link |
|-----------|-------------------------------------|------|
| CNOT      | Porta NOT controlada                | [README](README_CNOT_PT.md) |
| CY        | Porta Y controlada                  | [README](README_ControlledY_PT.md) |
| CZ        | Porta Z controlada                  | [README](README_ControlledZ_PT.md) |
| SWAP      | Troca estados entre dois qubits     | [README](README_SWAP_PT.md) |
| iSWAP     | Troca com fase imaginária `i`       | [README](README_iSWAP_PT.md) |
| CU        | Porta unitária arbitrária controlada| [README](README_ControlledU_PT.md) |

---

## 🔹 Portas de 3 Qubits

| Porta     | Descrição                           | Link |
|-----------|-------------------------------------|------|
| Toffoli   | NOT controlada por dois qubits      | [README](README_Toffoli_PT.md) |
| Fredkin   | SWAP controlada                     | [README](README_Fredkin_PT.md) |