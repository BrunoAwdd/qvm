#include <cuComplex.h>

/// CUDA kernel to apply the Fredkin (Controlled-SWAP) gate.
///
/// This gate swaps the target qubits `t1` and `t2` **only when**
/// the control qubit is in the `|1⟩` state.
///
/// # Parameters
/// - `state`: Quantum state vector (cuDoubleComplex[], size 2^n)
/// - `n_qubits`: Total number of qubits in the system
/// - `control`: Index of the control qubit (0-based)
/// - `t1`, `t2`: Indices of the target qubits to be swapped
///
/// # Notes
/// - The condition `if (idx < swap_idx)` ensures only one thread per pair does the swap.
extern "C" __global__
void fredkin_kernel(
    cuDoubleComplex* state,
    int n_qubits,
    int control,
    int t1,
    int t2
) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;
    if (idx >= size) return;

    // Extract relevant qubit bits
    bool ctrl_bit = (idx >> control) & 1;
    bool b1 = (idx >> t1) & 1;
    bool b2 = (idx >> t2) & 1;

    // If control is 1 and t1 and t2 differ, swap amplitudes
    if (ctrl_bit && b1 != b2) {
        unsigned long long int swap_idx = idx ^ ((1ULL << t1) | (1ULL << t2));
        if (idx < swap_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[swap_idx];
            state[swap_idx] = temp;
        }
    }
}
