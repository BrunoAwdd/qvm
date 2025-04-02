#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void ry_kernel(cuDoubleComplex* state, int qubit, int num_qubits, double theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);

        // RY = real-valued
        state[i].x = c * a.x - s * b.x;
        state[i].y = c * a.y - s * b.y;

        state[partner].x = s * a.x + c * b.x;
        state[partner].y = s * a.y + c * b.y;
    }
}
