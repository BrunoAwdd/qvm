import sys
import os
from math import pi

# Adiciona o diretório base ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from qlang.qlang import QLangScript

def test_u1_gate(qubits=1, device="cpu"):
    q = QLangScript(device)
    q.create(qubits)

    q.u1(0, pi)  # aplica um RZ(pi), equivalente a um phase(π)
    q.h(0)       # aplica Hadamard
    q.m()        # mede o qubit
    q.run()

    result = q.get_measurement_result()
    print("Resultado da medição com U1(π):", result[0])
    
    # Verifica se o resultado está entre 0 ou 1 (colapso válido)
    assert result[0] in [0, 1], "Medição inválida após aplicação do U1"
