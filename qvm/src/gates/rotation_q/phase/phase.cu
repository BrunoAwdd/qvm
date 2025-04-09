#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void phase_kernel(cuDoubleComplex* state, int qubit, int num_qubits, double theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        cuDoubleComplex phase;
        phase.x = cos(theta);  
        phase.y = sin(theta); 

        cuDoubleComplex v = state[i];
        state[i] = cuCmul(v, phase);
    }
}
