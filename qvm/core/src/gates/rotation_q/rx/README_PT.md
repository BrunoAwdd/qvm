# 🔄 Porta `RX(θ)` – Simulação Quântica QLang

A porta `RX` realiza uma **rotação em torno do eixo X** da esfera de Bloch por um ângulo `θ` (em radianos). É uma porta paramétrica de um único qubit, amplamente usada em algoritmos quânticos e circuitos variacionais.

---

## 📐 Representação Matricial

```
RX(θ) = cos(θ/2)·I − i·sin(θ/2)·X

      =
      [ cos(θ/2)    -i·sin(θ/2) ]
      [ -i·sin(θ/2)  cos(θ/2)   ]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::rx::RX;

let theta = std::f64::consts::PI / 2.0;
let rx = RX::new(theta);
let matrix = rx.matrix();
```

---

## 🧪 Testes

- `test_rx_gate_matrix_pi_2` – Valida a matriz para θ = π/2.
- `test_rx_gate_name` – Confirma que o nome da porta é `"RX"`.

---

## 📎 Notas

- Porta paramétrica (depende de θ).
- Presente em redes neurais quânticas e algoritmos variacionais.
- Quando `θ = π`, a RX se comporta como a Pauli-X.