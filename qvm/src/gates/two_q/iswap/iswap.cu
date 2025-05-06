#include <cuComplex.h>

/// CUDA kernel to apply the iSWAP gate to a pair of qubits.
///
/// The iSWAP gate swaps the amplitudes of `|01⟩` and `|10⟩`,
/// and multiplies them by `i`.
///
/// Matrix (4×4):
/// ```text
/// [ 1  0  0   0 ]
/// [ 0  0  i   0 ]
/// [ 0  i  0   0 ]
/// [ 0  0  0   1 ]
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (length 2^num_qubits)
/// - `q0`, `q1`: The two qubit indices the iSWAP is applied to
/// - `num_qubits`: Total number of qubits
///
/// # Behavior
/// For each state where `q0 != q1`, flips both bits and applies `i` phase.
extern "C" __global__
void iswap_kernel(
    cuDoubleComplex* state,
    int q0,
    int q1,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Extract individual bits
    int bit_q0 = (i >> q0) & 1;
    int bit_q1 = (i >> q1) & 1;

    // Apply iSWAP only to states |01⟩ and |10⟩
    if (bit_q0 != bit_q1) {
        int target = i ^ ((1 << q0) | (1 << q1));  // Flip both bits

        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[target];

        // Multiply by i: (a.x + i·a.y) → i·a = -a.y + i·a.x
        cuDoubleComplex i_a = make_cuDoubleComplex(-a.y, a.x);
        cuDoubleComplex i_b = make_cuDoubleComplex(-b.y, b.x);

        state[i] = i_b;
        state[target] = i_a;
    }
}
