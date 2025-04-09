import sys
import os
from math import pi

# Adiciona o diretório raiz ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from qlang.qlang import QLangScript

def test_u2_gate(qubits=1, device="cpu"):
    q = QLangScript(device)
    q.create(qubits)

    q0 = 0
    q1 = 0

    for _ in range(100):
        q.reset()
        q.u2(0, 0, pi)  # U2(0, π)
        #q.m()
        q.run()
        result = q.get_measurement_result()
        if result[0] == 1:
            q1 += 1
        else:
            q0 += 1

    print(f"Total de 0s: {q0}, Total de 1s: {q1}")
    
    # Verifica que há distribuição (espera-se que ambos apareçam)
    assert q0 > 0 and q1 > 0, "Esperava colapsos diferentes em U2(0, π)"

    # Verifica se está próximo de 50/50 com margem aceitável
    assert 30 < q0 < 70, f"Distribuição inesperada para 0s: {q0}"
    assert 30 < q1 < 70, f"Distribuição inesperada para 1s: {q1}"
