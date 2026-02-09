#include <cuComplex.h>

/// CUDA kernel to apply the CY (Controlled-Y) gate to a quantum state.
///
/// CY applies the Pauli-Y gate to the `target` qubit if the `control` qubit is 1.
///
/// Matrix:
/// [1 0 0 0]
/// [0 1 0 0]
/// [0 0 0 -i]
/// [0 0 i  0]
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `control`: Index of the control qubit (0-based)
/// - `target`: Index of the target qubit (0-based)
/// - `num_qubits`: Total number of qubits
extern "C" __global__
void cy_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if (((i >> control) & 1) == 1) {
        int flipped = i ^ (1 << target);

        // Operar apenas uma vez por par
        if (i < flipped) {
            cuDoubleComplex a = state[i];
            cuDoubleComplex b = state[flipped];

            // Y = [0 -i; i 0]
            state[i].x      =  b.y;
            state[i].y      = -b.x;

            state[flipped].x = -a.y;
            state[flipped].y =  a.x;
        }
    }
}
