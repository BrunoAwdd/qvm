import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


from qlang.qlang import QLangScript
from math import pi

def test_u3_gate(qubits=1, device="cpu"):
    q = QLangScript(device)
    q.create(qubits)
    q0 = 0
    q1 = 0
    for i in range(100):
        q.reset()
        # Equivalente a U2(0, π): U3(π/2, 0, π)
        q.u3(0, pi / 2, 0.0, pi)
        q.m()
        q.run()
        result = q.get_measurement_result()
        print("Resultado da medição com U3(π/2, 0, π):", result[0])
        if result[0] == 1:
            q1 += 1
        else:
            q0 += 1

    print(f"q0: {q0}% - q1: {q1}%")

def test_u3_gate_distribution():
    q = QLangScript("cpu")
    q.create(1)

    q0 = 0
    q1 = 0

    for _ in range(100):
        q.reset()
        q.u3(0, pi / 2, 0.0, pi)  # Equivalente a U2(0, π)
        #q.m()
        q.run()
        result = q.get_measurement_result()
        if result[0] == 1:
            q1 += 1
        else:
            q0 += 1

    print(f"q0: {q0} - q1: {q1}")

    assert q0 > 0 and q1 > 0, "Esperava colapsos diferentes com U3(π/2, 0, π)"
    assert 30 < q0 < 70, f"Distribuição inesperada para 0s: {q0}"
    assert 30 < q1 < 70, f"Distribuição inesperada para 1s: {q1}"
