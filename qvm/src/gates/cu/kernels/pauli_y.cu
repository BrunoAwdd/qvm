#include <cuComplex.h>

extern "C" __global__ void pauli_y_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        // Apply Y gate
        state[i].x =  b.y;
        state[i].y = -b.x;

        state[partner].x = -a.y;
        state[partner].y =  a.x;
    }
}
