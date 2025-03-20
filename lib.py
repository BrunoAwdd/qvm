import ctypes

class QuantumVM:
    def __init__(self, num_qubits: int):
        # Carregando a biblioteca compartilhada compilada pelo Rust
        self.qvm_lib = ctypes.CDLL('./qvm/target/release/libqvm.so')  # Caminho para a biblioteca
        self.num_qubits = num_qubits
        # Inicializando a QVM com o número de qubits
        self.qvm_ptr = self.create_qvm(num_qubits)

    def create_qvm(self, num_qubits: int):
        """Cria a QVM com um dado número de qubits."""
        self.qvm_lib.create_qvm.argtypes = [ctypes.c_size_t]
        self.qvm_lib.create_qvm.restype = ctypes.POINTER(ctypes.c_void_p)
        return self.qvm_lib.create_qvm(num_qubits)

    def apply_hadamard(self, qubit: int):
        """Aplica o gate Hadamard ao qubit especificado."""
        self.qvm_lib.apply_hadamard.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.qvm_lib.apply_hadamard.restype = None
        self.qvm_lib.apply_hadamard(self.qvm_ptr, qubit)

    def apply_pauli_x(self, qubit: int):
        """Aplica o gate Pauli-X ao qubit especificado."""
        self.qvm_lib.apply_pauli_x.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.qvm_lib.apply_pauli_x.restype = None
        self.qvm_lib.apply_pauli_x(self.qvm_ptr, qubit)

    def apply_cnot(self, control_qubit: int, target_qubit: int):
        """Aplica o gate CNOT entre os qubits de controle e alvo."""
        self.qvm_lib.apply_cnot.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
        self.qvm_lib.apply_cnot.restype = None
        self.qvm_lib.apply_cnot(self.qvm_ptr, control_qubit, target_qubit)

    def display_qvm(self):
        """Exibe o estado atual do sistema quântico."""
        self.qvm_lib.display_qvm.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.qvm_lib.display_qvm.restype = None
        self.qvm_lib.display_qvm(self.qvm_ptr)

    def measure_all(self):
        """Mede todos os qubits e retorna os resultados corretamente."""
        
        # Definir o tipo de retorno como um ponteiro para um array de `c_uint8`
        self.qvm_lib.measure_all.restype = ctypes.POINTER(ctypes.c_uint8)

        # Chamar a função e obter um ponteiro para um array de bytes
        result_ptr = self.qvm_lib.measure_all(self.qvm_ptr)
        
        if not result_ptr:
            print("Erro: Ponteiro retornado é NULL.")
            return []

        # Converter o ponteiro retornado em uma lista Python de inteiros
        result_list = [result_ptr[i] for i in range(self.num_qubits)]

        return result_list

    def free_qvm(self):
        """Libera a memória alocada para a QVM."""
        self.qvm_lib.free_qvm.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.qvm_lib.free_qvm.restype = None
        self.qvm_lib.free_qvm(self.qvm_ptr)

if __name__ == "__main__":
    # Inicializa a QVM com 2 qubits
    qvm = QuantumVM(3)

    # Aplica uma sequência de operações
    print("Aplicando Hadamard no qubit 0:")
    qvm.apply_hadamard(0)
    qvm.display_qvm()

    print("Aplicando Pauli-X no qubit 1:")
    qvm.apply_pauli_x(1)
    qvm.display_qvm()

    print("Aplicando CNOT nos qubits 0 e 1:")
    qvm.apply_cnot(0, 1)
    qvm.display_qvm()

    print("Medindo todos os qubits:")
    result = qvm.measure_all()
    print(f"Resultado da medição: {result}")

    # Liberando a memória alocada
    qvm.free_qvm()