#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the Hadamard (H) gate to a specific qubit.
///
/// The Hadamard gate puts a qubit into superposition by performing:
///   H = (1/√2) * | 1  1 |
///                | 1 -1 |
///
/// This kernel updates the quantum state vector in-place by computing the linear
/// combination of amplitudes that differ only in the target qubit.
///
/// # Parameters
/// - `state`: Quantum state vector as array of cuDoubleComplex (length 2^n)
/// - `qubit`: The index of the qubit to apply the gate to
/// - `num_qubits`: Total number of qubits in the system
///
/// # Note
/// - To avoid double swaps, only threads with `i < partner` perform the update.
extern "C" __global__ void hadamard_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // Find partner index (same as i, but with the target qubit flipped)
    int partner = i ^ (1 << qubit);

    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        double sqrt2_inv = 0.70710678118;  // Approximate 1 / sqrt(2)

        // H|a,b⟩ = (a ± b) / √2
        state[i].x = sqrt2_inv * (a.x + b.x);
        state[i].y = sqrt2_inv * (a.y + b.y);

        state[partner].x = sqrt2_inv * (a.x - b.x);
        state[partner].y = sqrt2_inv * (a.y - b.y);
    }
}
