from qlang import QLangScript
from math import pi

def test_u2_gate():
    q = QLangScript("cuda")
    q.create(1)
    q0 = 0
    q1 = 0
    for i in range (100):
        q.reset()
        q.u2(0, 0, pi)  # φ = 0, λ = π — equivale a um U3(π/2, 0, π)
        q.m()
        q.run()
        result = q.get_measurement_result()
        print(f"Resultado da medição com U2(0, π){1}:", result[0])
        if result[0] == 1:
            q1 = q1 + 1
        else:
            q0 = q0 + 1

    print(f"q0: {q0}% - q1: {q1}%")

test_u2_gate()
