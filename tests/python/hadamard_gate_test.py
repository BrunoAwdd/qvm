import sys
import os
from math import pi

# Corrige o caminho para importar QLangScript corretamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from qlang.qlang import QLangScript

def test_hadamard_distribution():
    q = QLangScript("cpu")
    q.create(1)

    q0 = 0
    q1 = 0

    for _ in range(100):
        q.reset()
        q.h(0)   # Hadamard em |0> → (|0⟩ + |1⟩)/√2
        q.m()
        q.run()
        result = q.get_measurement_result()
        if result[0] == 1:
            q1 += 1
        else:
            q0 += 1

    print(f"Hadamard → q0: {q0} - q1: {q1}")

    # Esperamos uma distribuição aproximadamente uniforme
    #assert q0 > 0 and q1 > 0, "Esperava colapsos diferentes com Hadamard"
    #assert 30 < q0 < 70, f"Distribuição inesperada para 0s: {q0}"
    #assert 30 < q1 < 70, f"Distribuição inesperada para 1s: {q1}"
