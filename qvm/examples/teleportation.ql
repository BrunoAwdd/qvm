// Quantum Teleportation Protocol
// Teleporting state from q0 to q2 using q1 as helper.

create(3)

// 1. Prepare state to teleport on q0 (e.g., |+>)
h(0)
rz(0, 1.57) // Some phase

// 2. Create Bell pair (EPR) between q1 and q2
h(1)
cnot(1, 2)

// 3. Bell measurement on q0 and q1
cnot(0, 1)
h(0)

let m0 = measure(0)
let m1 = measure(1)

// 4. Apply corrections on q2
if (m1) {
    x(2)
}
if (m0) {
    z(2)
}

// 5. Verification
// q2 should now be in the state q0 was in initially.
// We apply inverse operations to q2 to check if it returns to |0>
rz(2, -1.57)
h(2)

// If successful, q2 should be |0>
display()
measure(2)
