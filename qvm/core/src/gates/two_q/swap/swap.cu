#include <cuComplex.h>

/// CUDA kernel to apply a SWAP gate between two qubits.
///
/// The SWAP gate exchanges the amplitudes of states where the bits `q1` and `q2` differ.
/// It swaps the corresponding pairs of basis states:
/// - |...0...1...⟩ ⟷ |...1...0...⟩
///
/// # Parameters
/// - `state`: Quantum state vector (cuDoubleComplex[]), of size 2^n
/// - `n_qubits`: Total number of qubits
/// - `q1`, `q2`: Indices of the two qubits to swap
///
/// # Behavior
/// - If bits at positions `q1` and `q2` differ, their amplitudes are exchanged.
/// - Each swap is done only once (for idx < swap_idx).
extern "C" __global__
void apply_swap(
    cuDoubleComplex* state,
    int n_qubits,
    int q1,
    int q2
) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int size = 1ULL << n_qubits;
    if (idx >= size) return;

    bool bit_q1 = (idx >> q1) & 1;
    bool bit_q2 = (idx >> q2) & 1;

    if (bit_q1 != bit_q2) {
        // Flip bits q1 and q2 to get the swapped state index
        unsigned long long int swap_idx = idx ^ ((1ULL << q1) | (1ULL << q2));

        // Perform swap only once
        if (idx < swap_idx) {
            cuDoubleComplex temp = state[idx];
            state[idx] = state[swap_idx];
            state[swap_idx] = temp;
        }
    }
}
