#include <cuComplex.h>

extern "C" __global__ void pauli_x_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex temp = state[i];
        state[i] = state[partner];
        state[partner] = temp;
    }
}
