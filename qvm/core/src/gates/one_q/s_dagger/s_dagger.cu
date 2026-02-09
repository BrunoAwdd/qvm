#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the S† (S-dagger) gate to a target qubit.
///
/// The S† gate is the inverse of the S gate. It applies a −π/2 phase shift (i.e., multiplies by `-i`)
/// to the `|1⟩` component of the target qubit:
///
/// ```text
/// | 1    0 |
/// | 0  -i |
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (`cuDoubleComplex*`, length 2^n)
/// - `target`: The target qubit index (0-based)
/// - `num_qubits`: Total number of qubits in the system
///
/// # Behavior
/// - For each amplitude where the `target` qubit is `1`, multiply it by `-i`
extern "C" __global__ void sdagger_kernel(
    cuDoubleComplex* state,
    int target,
    int num_qubits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;

    if (idx >= size) return;

    int mask = 1 << target;
    if ((idx & mask) != 0) {
        cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0); // -i
        state[idx] = cuCmul(minus_i, state[idx]);
    }
}
