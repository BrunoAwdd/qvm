#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void hadamard_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        double sqrt2_inv = 0.70710678118;

        state[i].x = sqrt2_inv * (a.x + b.x);
        state[i].y = sqrt2_inv * (a.y + b.y);

        state[partner].x = sqrt2_inv * (a.x - b.x);
        state[partner].y = sqrt2_inv * (a.y - b.y);
    }
}
