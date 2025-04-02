#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void t_kernel(cuDoubleComplex* state, int qubit, int num_qubits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        double angle = M_PI / 4.0;
        cuDoubleComplex phase = make_cuDoubleComplex(cos(angle), sin(angle)); // e^{iπ/4}
        state[i] = cuCmul(state[i], phase);
    }
}
