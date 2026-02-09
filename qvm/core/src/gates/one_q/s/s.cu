#include <cuComplex.h>

/// CUDA kernel to apply the S gate (Phase gate) to a specific qubit.
///
/// The S gate applies a π/2 phase (multiplication by `i`) to the `|1⟩` component
/// of the target qubit. The matrix is:
///
/// ```text
/// | 1   0 |
/// | 0   i |
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (array of `cuDoubleComplex`)
/// - `qubit`: The index of the qubit to apply the S gate to
/// - `num_qubits`: Total number of qubits in the system
///
/// # Behavior
/// - For each amplitude in the state, if the target qubit is `1`, multiply it by `i`.
extern "C" __global__ void s_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Check if qubit is in |1⟩ state
    if ((i >> qubit) & 1) {
        cuDoubleComplex phase = make_cuDoubleComplex(0.0, 1.0); // i
        state[i] = cuCmul(state[i], phase);
    }
}
