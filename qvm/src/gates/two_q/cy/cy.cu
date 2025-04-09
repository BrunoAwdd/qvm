#include <cuComplex.h>

extern "C" __global__ void cy_kernel(cuDoubleComplex* state, int control, int target, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int control_bit = (i >> control) & 1;

    // Para CY, aplicamos Y se control == 1
    if (control_bit == 1) {
        int flipped = i ^ (1 << target);

        cuDoubleComplex partner = state[flipped];

        double sign = ((i >> target) & 1) == 0 ? -1.0 : 1.0;

        state[i].x =  sign * partner.y;
        state[i].y = -sign * partner.x;
    }
}
