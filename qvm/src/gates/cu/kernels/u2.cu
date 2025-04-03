#include <cuComplex.h>
#include <math.h>

extern "C"
__global__ void u2_kernel(cuDoubleComplex* state, int target, int num_qubits, double phi, double lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;
    if (idx >= size) return;

    int mask = 1 << target;

    if ((idx & mask) == 0) {
        int pair = idx | mask;

        cuDoubleComplex a = state[idx];
        cuDoubleComplex b = state[pair];

        // Pré-calcula fases
        cuDoubleComplex e_il   = make_cuDoubleComplex(cos(lambda), sin(lambda));
        cuDoubleComplex e_ip   = make_cuDoubleComplex(cos(phi), sin(phi));
        cuDoubleComplex e_ipl  = make_cuDoubleComplex(cos(phi + lambda), sin(phi + lambda));
        double inv_sqrt2 = 1.0 / sqrt(2.0);
        cuDoubleComplex c = make_cuDoubleComplex(inv_sqrt2, 0.0);

        cuDoubleComplex new_a = cuCsub(cuCmul(c, a), cuCmul(cuCmul(c, e_il), b));
        cuDoubleComplex new_b = cuCadd(cuCmul(cuCmul(c, e_ip), a), cuCmul(cuCmul(c, e_ipl), b));

        state[idx] = new_a;
        state[pair] = new_b;
    }
}
