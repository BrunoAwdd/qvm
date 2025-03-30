import ctypes

class QLangScript:
    def __init__(self, lib_path="./qvm/target/release/libqvm.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.code_lines = []

        self.lib.run_qlang_inline.argtypes = [ctypes.c_char_p]
        self.lib.run_qlang_inline.restype = None

        self.lib.reset_qvm.restype = None
        self.lib.run_qlang.restype = None
        self.lib.measure_all.restype = ctypes.POINTER(ctypes.c_uint8)  # Expecta-se um ponteiro para array de bytes
        self.lib.display_qvm.restype = ctypes.POINTER(ctypes.c_char)  # Expecta-se um ponteiro para a string

    def line(self, s):
        self.code_lines.append(s)

    def create(self, n):
        self.lib.create_qvm.argtypes = [ctypes.c_size_t]
        self.lib.create_qvm.restype = None
        self.lib.create_qvm(n)

    def hadamard(self, q): self.line(f"hadamard({q})")
    def h(self, q): self.hadamard(q)

    def pauli_x(self, q): self.line(f"paulix({q})")
    def x(self, q): self.pauli_x(q)

    def pauli_y(self, q): self.line(f"pauliy({q})")
    def y(self, q): self.pauli_y(q)

    def pauli_z(self, q): self.line(f"pauliz({q})")
    def z(self, q): self.pauli_z(q)

    def cnot(self, c, t): self.line(f"cnot({c},{t})")
    def cx(self, c, t): self.cnot(c, t)

    def measure_all(self): self.line("measure_all()")
    def m(self): self.measure_all()

    def display(self): self.line("display()")
    def d(self): self.display()

    def run(self):
        code = "\n".join(self.code_lines)
        code_bytes = code.encode("utf-8")
        self.lib.run_qlang_inline(ctypes.c_char_p(code_bytes))

    def reset(self):
        self.code_lines = []
        self.lib.reset_qvm()

    def get_measurement_result(self):
        """Chama measure_all e retorna o resultado"""
        result_ptr = self.lib.measure_all()
        if result_ptr:
            result = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_uint8))
            return [result[i] for i in range(0, self.get_num_qubits())]
        return []

    def get_qvm_state(self):
        """Chama display_qvm e retorna o estado do QVM"""
        result_ptr = self.lib.display_qvm()
        if result_ptr:
            state_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode("utf-8")
            return state_str
        return ""

    def get_num_qubits(self):
        """Obtém o número de qubits"""
        self.lib.get_num_qubits.restype = ctypes.c_size_t
        return self.lib.get_num_qubits()
    

if __name__ == "__main__":
    q = QLangScript()
    q.create(1)

    for i in range(10):
        q.reset()
        q.h(0)
        q.m()
        q.run()
        print(f"Resultado da medição [{i}]:", q.get_measurement_result())
