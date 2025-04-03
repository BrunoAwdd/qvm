#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void tdagger_kernel(cuDoubleComplex* state, int target, int num_qubits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;
    if (idx >= size) return;

    if ((idx >> target) & 1) {
        double angle = -M_PI / 4.0;
        cuDoubleComplex phase = make_cuDoubleComplex(cos(angle), sin(angle));
        state[idx] = cuCmul(state[idx], phase);
    }
}
