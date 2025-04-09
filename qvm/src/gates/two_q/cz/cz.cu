#include <cuComplex.h>

extern "C" __global__ void cz_kernel(cuDoubleComplex* state, int control, int target, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int control_bit = (i >> control) & 1;
    int target_bit = (i >> target) & 1;

    if (control_bit == 1 && target_bit == 1) {
        state[i].x = -state[i].x;
        state[i].y = -state[i].y;
    }
}
