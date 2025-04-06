# 🌀 Supported Gates in QLang/QVM

This document lists all quantum gates available (or planned) in the **QLang** language and the **QVM** simulator, with quick explanations, backend coverage (CPU / CUDA), and implementation status.

---

### 📊 **Summary**

- **Implemented Gates:** `21`
- **Planned Gates:** `6`
- **Total Planned:** `27`
- **Backends:** `✅ CPU` — `✅ CUDA`

---

## 🔹 1. Single-Qubit Gates

| Name     | Code   | Function                      | Status | CPU | CUDA |
| -------- | ------ | ----------------------------- | ------ | --- | ---- |
| Pauli-X  | x(q)   | Flips the state: 0⟩ ↔ 1⟩      | ✅     | ✅  | ✅   |
| Pauli-Y  | y(q)   | Rotation with imaginary phase | ✅     | ✅  | ✅   |
| Pauli-Z  | z(q)   | Applies -1 phase to 1⟩        | ✅     | ✅  | ✅   |
| Hadamard | h(q)   | Superposition (0⟩ +1⟩)/√2     | ✅     | ✅  | ✅   |
| S        | s(q)   | Applies phase of π/2          | ✅     | ✅  | ✅   |
| S-dagger | sdg(q) | Inverse of S                  | ✅     | ✅  | ✅   |
| T        | t(q)   | Applies phase of π/4          | ✅     | ✅  | ✅   |
| T-dagger | tdg(q) | Inverse of T                  | ✅     | ✅  | ✅   |
| Identity | id(q)  | Does nothing                  | ✅     | ✅  | ✅   |

---

## 🔸 2. Rotation Gates

| Name  | Code           | Function                    | Status | CPU | CUDA |
| ----- | -------------- | --------------------------- | ------ | --- | ---- |
| Rx    | rx(θ, q)       | Rotation on X-axis          | ✅     | ✅  | ✅   |
| Ry    | ry(θ, q)       | Rotation on Y-axis          | ✅     | ✅  | ✅   |
| Rz    | rz(θ, q)       | Rotation on Z-axis          | ✅     | ✅  | ✅   |
| U3    | u3(θ, φ, λ, q) | Universal single-qubit gate | ✅     | ✅  | ✅   |
| U2    | u2(φ, λ, q)    | Midpoint between H and U3   | ✅     | ✅  | ✅   |
| U1    | u1(λ, q)       | Scaled Rz gate              | ✅     | ✅  | ✅   |
| Phase | phase(θ, q)    | Applies arbitrary phase     | ⬜     | ⬜  | ⬜   |

---

## 🔻 3. Two-Qubit Gates (Controlled)

| Name  | Code       | Function                             | Status | CPU | CUDA |
| ----- | ---------- | ------------------------------------ | ------ | --- | ---- |
| CNOT  | cnot(c, t) | Applies X on target if control is 1⟩ | ✅     | ✅  | ✅   |
| CZ    | cz(c, t)   | Applies Z on target if control is 1⟩ | ⬜     | ⬜  | ⬜   |
| CY    | cy(c, t)   | Applies Y on target if control is 1⟩ | ⬜     | ⬜  | ⬜   |
| SWAP  | swap(a, b) | Swaps the states of two qubits       | ✅     | ✅  | ✅   |
| iSWAP | iswap(a,b) | Swap with imaginary phase            | ⬜     | ⬜  | ⬜   |

---

## 🔺 4. Three-Qubit Gates

| Name    | Code             | Function                        | Status | CPU | CUDA |
| ------- | ---------------- | ------------------------------- | ------ | --- | ---- | --- |
| Toffoli | toffoli(c1,c2,t) | Applies X if both controls are  | 1⟩     | ✅  | ✅   | ✅  |
| Fredkin | fredkin(c,q1,q2) | Swaps two targets if control is | 1⟩     | ✅  | ✅   | ✅  |

---

## 🎯 5. Measurements

| Name      | Code      | Function                            | Status                         | CPU | CUDA |
| --------- | --------- | ----------------------------------- | ------------------------------ | --- | ---- |
| Z Measure | m(q) / mz | Measures in Z computational basis   | ✅ Implemented                 | ✅  | ✅   |
| X Measure | mx(q)     | Applies H + measures (X basis)      | ⬜ Planned (via Python helper) | ⬜  | ⬜   |
| Y Measure | my(q)     | Applies S† + H + measures (Y basis) | ⬜ Planned (via Python helper) | ⬜  | ⬜   |

---

## 🧩 Legend

- ✅ _Implemented_
- ⬜ _Planned_
- CPU / CUDA indicate backend support

---

## 🤝 Contributing

To contribute with new gates, improvements or optimizations:

- Clone the repo (`git clone ...`)
- Check the `CONTRIBUTING.md` file
- Or open an _issue_ with your suggestion or question
