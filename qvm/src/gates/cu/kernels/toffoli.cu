#include <cuComplex.h>

extern "C" __global__
void toffoli_kernel(cuDoubleComplex* state, int n_qubits, int c1, int c2, int target) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;
    if (idx >= size) return;

    bool b1 = (idx >> c1) & 1;
    bool b2 = (idx >> c2) & 1;
    if (b1 && b2) {
        unsigned long long int flip_mask = 1ULL << target;
        unsigned long long int pair_idx = idx ^ flip_mask;
        if (idx < pair_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[pair_idx];
            state[pair_idx] = temp;
        }
    }
}
