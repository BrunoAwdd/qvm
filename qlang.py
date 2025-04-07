import ctypes
import os

class QLangScript:
    def __init__(self, backend="cpu"):
        backend = backend.lower()
        if backend not in ["cpu", "cuda"]:
            raise ValueError("Backend deve ser 'cpu' ou 'cuda'")
        
        lib_name = f"libqlang_{backend}.so"
        lib_path = os.path.abspath(f"./qvm/target/release/{lib_name}")
        self.lib = ctypes.CDLL(lib_path)
        self.code_lines = []
        self.backend = backend

        # Protótipos
        self.lib.run_qlang_inline.argtypes = [ctypes.c_char_p]
        self.lib.run_qlang_inline.restype = None
        self.lib.reset_qvm.restype = None
        self.lib.run_qlang.restype = None
        self.lib.measure_all.restype = ctypes.POINTER(ctypes.c_uint8)
        self.lib.display_qvm.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.get_num_qubits.restype = ctypes.c_size_t

    # Utils
    def line(self, s): self.code_lines.append(s)
    def reset(self): self.code_lines = []; self.lib.reset_qvm()
    def run(self):
        code = "\n".join(self.code_lines).encode("utf-8")
        self.lib.run_qlang_inline(ctypes.c_char_p(code))
        
    def get_num_qubits(self): return self.lib.get_num_qubits()

    def assert_qubit_range(self, *qs):
        n = self.get_num_qubits()
        for q in qs:
            if q >= n:
                raise ValueError(f"Qubit fora do range (0..{n-1}): {q}")

    # QVM Setup
    def create(self, n):
        self.lib.create_qvm.argtypes = [ctypes.c_size_t]
        self.lib.create_qvm(n)
        self.line(f"create({n})")

    # Gates de 1 qubit
    def identity(self, q): self.assert_qubit_range(q); self.line(f"id({q})")
    def hadamard(self, q): self.assert_qubit_range(q); self.line(f"hadamard({q})")
    def pauli_x(self, q):  self.assert_qubit_range(q); self.line(f"paulix({q})")
    def pauli_y(self, q):  self.assert_qubit_range(q); self.line(f"pauliy({q})")
    def pauli_z(self, q):  self.assert_qubit_range(q); self.line(f"pauliz({q})")
    def s(self, q):        self.assert_qubit_range(q); self.line(f"s({q})")
    def s_dagger(self, q): self.assert_qubit_range(q); self.line(f"sdagger({q})")
    def t(self, q):        self.assert_qubit_range(q); self.line(f"t({q})")
    def t_dagger(self, q): self.assert_qubit_range(q); self.line(f"tdagger({q})")
    def u1(self, q, lambda_): self.assert_qubit_range(q); self.line(f"u1({q},{lambda_})")
    def u2(self, q, phi, lambda_): self.assert_qubit_range(q); self.line(f"u2({q},{phi},{lambda_})")
    def u3(self, q, theta, phi, lambda_): self.assert_qubit_range(q); self.line(f"u3({q},{theta},{phi},{lambda_})")

    # Aliases
    def id(self, q): self.identity(q)
    def h(self, q): self.hadamard(q)
    def x(self, q): self.pauli_x(q)
    def y(self, q): self.pauli_y(q)
    def z(self, q): self.pauli_z(q)
    def m(self):    self.measure_all()
    def d(self):    self.display()
    def sdg(self, q): self.s_dagger(q)
    def tdg(self, q): self.t_dagger(q)
    def cx(self, c, t): self.cnot(c, t)

    # Gates de 2 qubits
    def cnot(self, c, t):
        self.assert_qubit_range(c, t)
        if c == t:
            raise ValueError(f"CNOT requer qubits distintos: {c} == {t}")
        self.line(f"cnot({c},{t})")

    def swap(self, c, t):
        self.assert_qubit_range(c, t)
        if c == t:
            raise ValueError(f"SWAP requer qubits distintos: {c} == {t}")
        self.line(f"swap({c},{t})")

    def toffoli(self, c, t1, t2):
        self.assert_qubit_range(c, t1, t2)
        if c == t1 or c == t2:
            raise ValueError(f"Toffoli requer qubits distintos: {c} == {t1} ou {c} == {t2}")
        self.line(f"toffoli({c},{t1},{t2})")

    def fredkin(self, c, t1, t2):
        self.assert_qubit_range(c, t1, t2)
        if c == t1 or c == t2:
            raise ValueError(f"Fredkin requer qubits distintos: {c} == {t1} ou {c} == {t2}")
        self.line(f"fredkin({c},{t1},{t2})")

    # Gates de rotação
    def rx(self, q, theta): self.assert_qubit_range(q); self.line(f"rx({q},{theta})")
    def ry(self, q, theta): self.assert_qubit_range(q); self.line(f"ry({q},{theta})")
    def rz(self, q, theta): self.assert_qubit_range(q); self.line(f"rz({q},{theta})")

    def measure(self, *qubits):
        self.assert_qubit_range(*qubits)

        arr_type = ctypes.c_size_t * len(qubits)
        arr = arr_type(*qubits)

        self.lib.measure.restype = ctypes.POINTER(ctypes.c_size_t)
        raw = self.lib.measure(arr, len(qubits))

        result = {}
        for i in range(len(qubits)):
            q = raw[i * 2]
            v = raw[i * 2 + 1]
            result[q] = v

        return result


    # Ações
    def measure_all(self):
        code = "\n".join(self.code_lines).encode("utf-8")
        ptr = self.lib.measure_all(ctypes.c_char_p(code))
        self.code_lines = []  # opcional: limpa o buffer de código após executar
        return [ptr[i] for i in range(self.get_num_qubits())] if ptr else []
    
    def display(self): self.line("display()")

    # Resultados
    def get_measurement_result(self):
        ptr = self.lib.measure_all()
        return [ptr[i] for i in range(self.get_num_qubits())] if ptr else []

    def get_qvm_state(self):
        ptr = self.lib.display_qvm()
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8") if ptr else ""

    def print(self):
        print("\n".join(self.code_lines))
    
    def save(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))

    def __str__(self):
        return ctypes.cast(self.lib.get_qlang_source(), ctypes.c_char_p).value.decode("utf-8")
