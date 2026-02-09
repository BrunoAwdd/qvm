#include <cuComplex.h>

/// CUDA kernel to apply a controlled-U gate to a quantum state vector.
///
/// This kernel operates on a state vector represented as an array of `cuDoubleComplex` amplitudes,
/// and applies a 2x2 unitary matrix `U` to the target qubit **only if** the control qubit is in the `|1⟩` state.
///
/// # Parameters
/// - `state`: The quantum state vector (length 2^num_qubits).
/// - `control`: The index of the control qubit (0-based).
/// - `target`: The index of the target qubit (0-based).
/// - `num_qubits`: Total number of qubits in the system.
/// - `u00`, `u01`, `u10`, `u11`: Elements of the 2x2 unitary matrix to apply when control == 1.
///
/// # Note
/// - To avoid race conditions, only one thread in each control-target pair performs the update.
/// - This kernel assumes `target != control`.
extern "C" __global__ void controlled_u_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits,
    cuDoubleComplex u00,
    cuDoubleComplex u01,
    cuDoubleComplex u10,
    cuDoubleComplex u11
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Apply only if the control qubit is |1⟩
    if (((i >> control) & 1) == 1) {
        // Determine target bit and its pair (with flipped target bit)
        int bit_target = (i >> target) & 1;
        int pair = i ^ (1 << target); // Flip the target bit

        // Only one thread in each (i, pair) group proceeds
        if (i < pair) return;

        cuDoubleComplex amp_i    = state[i];
        cuDoubleComplex amp_pair = state[pair];

        // Apply matrix transformation
        if (bit_target == 0) {
            state[i]    = cuCadd(cuCmul(u00, amp_i), cuCmul(u01, amp_pair));
            state[pair] = cuCadd(cuCmul(u10, amp_i), cuCmul(u11, amp_pair));
        } else {
            state[i]    = cuCadd(cuCmul(u11, amp_i), cuCmul(u10, amp_pair));
            state[pair] = cuCadd(cuCmul(u01, amp_i), cuCmul(u00, amp_pair));
        }
    }
}
