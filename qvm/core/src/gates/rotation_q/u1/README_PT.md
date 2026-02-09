# 🧭 Porta `U1(λ)` – Simulação Quântica QLang

A porta `U1` é uma **porta de fase com um único parâmetro**, que aplica uma fase `λ` ao estado `|1⟩`, deixando o estado `|0⟩` inalterado. É equivalente a uma rotação `RZ(λ)` até uma fase global.

---

## 📐 Representação Matricial

```
U1(λ) =
[ 1       0     ]
[ 0   e^{iλ}    ]
```

---

## 🧰 Uso (Rust)

```rust
use qlang::gates::u1::U1;

let lambda = std::f64::consts::PI;
let u1 = U1::new(lambda);
let matrix = u1.matrix();
```

---

## 🧪 Testes (sugerido)

- Validar que `U1(λ)` equivale a `Phase(λ)` ou `RZ(λ)` até uma fase global.
- Confirmar que o nome da porta é `"U1"`.

---

## 📎 Notas

- A `U1` faz parte do conjunto de portas padrão do OpenQASM.
- É equivalente à `Phase(λ)` em várias implementações.