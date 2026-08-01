// Quantum-conditional hedge sizing across two sector-correlated assets.
//
// This is an infrastructure example, not a validated trading strategy: it
// demonstrates a primitive (`qif`) that lets portfolio logic react to a
// market signal that has not been resolved (measured) yet, without either
// collapsing that signal early or letting it be reused after it has already
// been spent. Classical branching can't do the first part; QLang's linear
// qubit typing enforces the second part at compile time (see
// `portfolio_hedge_reuse_error.ql`).

create(3)

// q0: market regime signal (0 = risk-off, 1 = risk-on), left in superposition.
// Hedge logic below reacts to both regimes coherently; which one "actually
// happened" for the portfolio is only decided at the final measurement.
let regime: qubit = alloc(0)
h(regime)

// q1, q2: hedge exposure for two sector-correlated assets, starting unhedged.
let asset_a: qubit = alloc(1)
let asset_b: qubit = alloc(2)

// Coherent conditional: flip asset_a's hedge only in the risk-on branch of
// `regime`, without measuring/collapsing `regime` first.
qif (0) {
    x(1)
}

// Sector correlation: asset_b's hedge tracks asset_a's hedge state.
cnot(1, 2)

// Resolve the scenario: sample one consistent (regime, hedge_a, hedge_b) outcome.
display()
measure_all()
