#include <cuComplex.h>

extern "C" __global__ void pauli_z_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        state[i].x = -state[i].x;
        state[i].y = -state[i].y;
    }
}
