from qlang import QLangScript
from math import pi

def test_u1_gate(qubits=1, device="cpu"):
    q = QLangScript(device)
    q.create(qubits)

    q.u1(0, pi)  # aplica um RZ(pi), com fator de fase

    q.h(0)       # aplica Hadamard
    q.m()        # mede o qubit
    q.run()

    result = q.get_measurement_result()
    print("Resultado da medição com U1(π):", result[0])
