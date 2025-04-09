#include <cuComplex.h>
extern "C" __global__
void fredkin_kernel(cuDoubleComplex* state, int n_qubits, int control, int t1, int t2) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;
    if (idx >= size) return;

    bool ctrl_bit = (idx >> control) & 1;
    bool b1 = (idx >> t1) & 1;
    bool b2 = (idx >> t2) & 1;

    if (ctrl_bit && b1 != b2) {
        unsigned long long int swap_idx = idx ^ ((1ULL << t1) | (1ULL << t2));
        if (idx < swap_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[swap_idx];
            state[swap_idx] = temp;
        }
    }
}
