// This program intentionally fails `qlang check`.
//
// It represents a realistic bug class in quantitative code: a risk signal
// gets resolved (measured) to size one hedge, then the same signal is reused
// as if it were still live to size a second, unrelated hedge. In a classical
// pipeline this kind of stale-signal reuse is a runtime/production bug. In
// QLang, `regime`'s linear qubit type marks it consumed at `measure`, so
// reusing it afterwards is rejected at compile time instead.

create(1)
let regime: qubit = alloc(0)
let resolved_regime: bit = measure(regime)
h(regime)
