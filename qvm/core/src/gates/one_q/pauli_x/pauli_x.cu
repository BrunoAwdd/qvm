#include <cuComplex.h>

/// CUDA kernel to apply the Pauli-X (NOT) gate to a single qubit.
///
/// This gate flips the target qubit: |0⟩ ↔ |1⟩. The operation swaps amplitude pairs
/// whose indices differ only in the bit corresponding to `qubit`.
///
/// # Parameters
/// - `state`: Pointer to the full quantum state vector (2^n complex amplitudes)
/// - `qubit`: Index (0-based) of the qubit to apply Pauli-X to
/// - `num_qubits`: Total number of qubits in the state
///
/// # Notes
/// - Only half of the threads (where `i < partner`) perform the swap to avoid double-swapping
extern "C" __global__ void pauli_x_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Compute index of the state to swap with (flipping the target qubit)
    int partner = i ^ (1 << qubit);

    // Only one thread per pair performs the swap
    if (i < partner) {
        cuDoubleComplex temp = state[i];
        state[i] = state[partner];
        state[partner] = temp;
    }
}
