#include <cuComplex.h>

extern "C" __global__ void apply_swap(cuDoubleComplex* state, int n_qubits, int q1, int q2) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;

    if (idx >= size) return;

    // Se os bits q1 e q2 são diferentes
    bool bit_q1 = (idx >> q1) & 1;
    bool bit_q2 = (idx >> q2) & 1;

    if (bit_q1 != bit_q2) {
        // Cria o índice alvo onde os bits q1 e q2 foram trocados
        unsigned long long int swap_idx = idx ^ ((1ULL << q1) | (1ULL << q2));

        // Evita duplo swap (idx < swap_idx)
        if (idx < swap_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[swap_idx];
            state[swap_idx] = temp;
        }
    }
}
