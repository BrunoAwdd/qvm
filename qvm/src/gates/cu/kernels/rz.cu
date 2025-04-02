#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void rz_kernel(cuDoubleComplex* state, int qubit, int num_qubits, double theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int bit = (i >> qubit) & 1;
    double angle = (bit == 0) ? -theta / 2.0 : theta / 2.0;

    cuDoubleComplex phase;
    phase.x = cos(angle);
    phase.y = sin(angle); 

    cuDoubleComplex v = state[i];
    state[i] = cuCmul(v, phase);
}
