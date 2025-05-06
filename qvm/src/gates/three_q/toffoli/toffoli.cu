#include <cuComplex.h>

/// CUDA kernel to apply the Toffoli (CCNOT) gate to a quantum state.
///
/// This 3-qubit gate flips the target qubit **only if** both control qubits are `1`.
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex), length 2^n
/// - `n_qubits`: Total number of qubits in the register
/// - `c1`: Index of the first control qubit
/// - `c2`: Index of the second control qubit
/// - `target`: Index of the target qubit
///
/// # Behavior
/// - For each `idx`, if both `c1` and `c2` are 1, then flip the target bit.
/// - Uses `idx < pair_idx` to ensure each swap is done only once.
extern "C" __global__
void toffoli_kernel(
    cuDoubleComplex* state,
    int n_qubits,
    int c1,
    int c2,
    int target
) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;
    if (idx >= size) return;

    bool b1 = (idx >> c1) & 1;
    bool b2 = (idx >> c2) & 1;

    if (b1 && b2) {
        unsigned long long int flip_mask = 1ULL << target;
        unsigned long long int pair_idx = idx ^ flip_mask;

        // Avoid duplicate swaps
        if (idx < pair_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[pair_idx];
            state[pair_idx] = temp;
        }
    }
}
