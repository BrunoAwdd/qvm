#include <cuComplex.h>

/// CUDA kernel to apply a CNOT (Controlled-NOT) gate to a quantum state vector.
///
/// This gate flips the target qubit only if the control qubit is `1`.
///
/// Matrix effect:
/// - |00⟩ → |00⟩
/// - |01⟩ → |01⟩
/// - |10⟩ → |11⟩
/// - |11⟩ → |10⟩
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `control`: Index of the control qubit (0-based)
/// - `target`: Index of the target qubit (0-based)
/// - `num_qubits`: Total number of qubits in the system
///
/// # Notes
/// - Uses `i < partner` to avoid double-swapping in amplitude pairs.
extern "C" __global__
void cnot_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // If control bit is 1
    if (((i >> control) & 1) == 1) {
        // Flip target bit
        int partner = i ^ (1 << target);

        // Swap only once
        if (i < partner) {
            cuDoubleComplex tmp = state[i];
            state[i] = state[partner];
            state[partner] = tmp;
        }
    }
}
