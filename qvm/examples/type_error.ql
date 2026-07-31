// This program intentionally fails `qlang check`: q0 is consumed by measure.
create(1)
let q0: qubit = alloc(0)
let result: bit = measure(q0)
h(q0)
