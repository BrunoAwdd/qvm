#include <cuComplex.h>

extern "C" __global__ void cnot_kernel(cuDoubleComplex* state, int control, int target, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if (((i >> control) & 1) == 1) {
        int partner = i ^ (1 << target);
        if (i < partner) {
            cuDoubleComplex tmp = state[i];
            state[i] = state[partner];
            state[partner] = tmp;
        }
    }
}
