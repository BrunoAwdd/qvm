#include <cuComplex.h>

extern "C" __global__ void s_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        cuDoubleComplex phase = make_cuDoubleComplex(0.0, 1.0); // i
        state[i] = cuCmul(state[i], phase);
    }
}
