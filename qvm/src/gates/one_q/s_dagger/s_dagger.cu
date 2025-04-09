#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void sdagger_kernel(cuDoubleComplex* state, int target, int num_qubits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;

    if (idx >= size) return;

    int mask = 1 << target;
    if ((idx & mask) != 0) {
        // Multiplica por -i = (0 - 1i)
        cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0);
        state[idx] = cuCmul(minus_i, state[idx]);
    }
}