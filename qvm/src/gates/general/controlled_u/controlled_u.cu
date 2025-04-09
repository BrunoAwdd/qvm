#include <cuComplex.h>

extern "C" __global__ void controlled_u_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits,
    cuDoubleComplex u00,
    cuDoubleComplex u01,
    cuDoubleComplex u10,
    cuDoubleComplex u11
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Só aplicar se controle == 1
    if (((i >> control) & 1) == 1) {
        int bit_target = (i >> target) & 1;

        int pair = i ^ (1 << target); // Flipa o bit alvo

        // Apenas metade dos índices acessa a troca para evitar duplicação
        if (i < pair) return;

        cuDoubleComplex amp_i = state[i];
        cuDoubleComplex amp_pair = state[pair];

        if (bit_target == 0) {
            state[i]     = cuCadd(cuCmul(u00, amp_i),     cuCmul(u01, amp_pair));
            state[pair]  = cuCadd(cuCmul(u10, amp_i),     cuCmul(u11, amp_pair));
        } else {
            state[i]     = cuCadd(cuCmul(u11, amp_i),     cuCmul(u10, amp_pair));
            state[pair]  = cuCadd(cuCmul(u01, amp_i),     cuCmul(u00, amp_pair));
        }
    }
}
