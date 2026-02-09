# 🔄 Porta `RY(θ)` – Simulação Quântica QLang

A porta `RY` realiza uma **rotação em torno do eixo Y** da esfera de Bloch por um ângulo `θ` (em radianos). É uma rotação de qubit com valores reais, amplamente usada em algoritmos variacionais e redes neurais quânticas.

---

## 📐 Representação Matricial

```
RY(θ) =
[  cos(θ/2)   -sin(θ/2) ]
[  sin(θ/2)    cos(θ/2) ]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::ry::RY;

let theta = std::f64::consts::PI / 2.0;
let ry = RY::new(theta);
let matrix = ry.matrix();
```

---

## 🧪 Testes

- `test_ry_matrix` – Valida a matriz gerada para θ = π/2.
- `test_ry_name` – Confirma que o nome da porta é `"RY"`.

---

## 📎 Notas

- Não envolve fases complexas; todos os valores são reais.
- Quando `θ = π`, a `RY` atua como uma Pauli-Y até uma fase global.
- Muito utilizada para criar superposições e controlar amplitudes.