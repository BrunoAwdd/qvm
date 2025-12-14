// QLang Standard Library

// Bell State: |00> + |11>
fn bell(q1, q2) {
    h(q1)
    cx(q1, q2)
}

// GHZ State: |000> + |111>
fn ghz(q1, q2, q3) {
    h(q1)
    cx(q1, q2)
    cx(q2, q3)
}

// Superposition: H on single qubit
fn superposition(q) {
    h(q)
}

// Aliases for 3-qubit gates
fn ccx(c1, c2, t) {
    toffoli(c1, c2, t)
}

fn cswap(c, t1, t2) {
    fredkin(c, t1, t2)
}

// Controlled-H (not native yet, keep it)
fn ch(c, t) {
    ry(t, 0.785398163) // pi/4
    cx(c, t)
    ry(t, -0.785398163) // -pi/4
}

// Teleportation Protocol
// src: qubit to teleport (must be prepared beforehand)
// anc: ancillary qubit (part of Bell pair)
// dest: destination qubit (part of Bell pair)
fn teleport(src, anc, dest) {
    // 1. Create Bell pair between anc and dest
    bell(anc, dest)
    
    // 2. Bell measurement on src and anc
    cx(src, anc)
    h(src)
    let m1 = measure(src)
    let m2 = measure(anc)
    
    // 3. Apply corrections
    if (m2 == 1) {
        x(dest)
    }
    if (m1 == 1) {
        z(dest)
    }
}
