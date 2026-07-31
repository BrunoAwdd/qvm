// Prepare and measure a three-qubit GHZ state.
create(3)
h(0)
cnot(0, 1)
cnot(1, 2)
measure_all()
