import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from qlang.qlang import QLangScript
import math

q = QLangScript(backend="cuda")  # ou "cpu"
q.create(1)

# Superposição no qubit 0
q.h(0)

# Aplica uma fase π/2
q.phase(0, math.pi / 2)

# Mede o qubit (irá colapsar com fase aplicada)
m = q.measure(0)

# Executa a simulação
q.run()

# Resultados
final = q.measure_all()
state = q.get_qvm_state()

print("📍 Medição após aplicação do phase(π/2):", m)
print("📍 Resultado final com measure_all():", final)
print("📄 Estado da QVM:")
print(state)
print("📄 Código QLang gerado:")
print(q.code_lines)
