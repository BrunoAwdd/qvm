#include <cuComplex.h>
#include <math.h>

extern "C" __global__
void u1_kernel(cuDoubleComplex* state, int target, int n, double lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << n;
    if (idx >= size) return;

    int mask = 1 << target;
    if (idx & mask) {
        cuDoubleComplex e_il = make_cuDoubleComplex(cos(lambda), sin(lambda));
        state[idx] = cuCmul(e_il, state[idx]);
    }
}
