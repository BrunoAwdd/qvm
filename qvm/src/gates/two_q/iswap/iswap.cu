#include <cuComplex.h>

extern "C" __global__ void iswap_kernel(cuDoubleComplex* state, int q0, int q1, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Verifica se os bits em q0 e q1 formam os estados |01⟩ ou |10⟩
    int bit_q0 = (i >> q0) & 1;
    int bit_q1 = (i >> q1) & 1;

    if (bit_q0 != bit_q1) {
        int target = i ^ ((1 << q0) | (1 << q1));  // Flip os dois bits

        // Pega os dois amplitudes envolvidos
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[target];

        // Multiplica por i
        cuDoubleComplex i_a = make_cuDoubleComplex(-a.y, a.x);
        cuDoubleComplex i_b = make_cuDoubleComplex(-b.y, b.x);

        state[i] = i_b;
        state[target] = i_a;
    }
}
