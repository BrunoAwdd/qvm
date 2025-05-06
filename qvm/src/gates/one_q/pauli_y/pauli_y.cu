#include <cuComplex.h>

/// CUDA kernel to apply the Pauli-Y gate to a specific qubit.
///
/// The Pauli-Y gate is defined by the matrix:
/// ```text
/// |  0  -i |
/// |  i   0 |
/// ```
///
/// It flips the target qubit like Pauli-X, but also applies a complex phase (±i).
///
/// # Parameters
/// - `state`: The quantum state vector (array of cuDoubleComplex values)
/// - `qubit`: The index (0-based) of the qubit to apply the gate to
/// - `num_qubits`: Total number of qubits in the quantum system
///
/// # Notes
/// - Only threads where `i < partner` perform the update to avoid duplicate swaps.
/// - The complex multiplication by `±i` is applied manually for performance.
extern "C" __global__ void pauli_y_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        // Apply Pauli-Y transformation:
        // i * a  = (-a.y, a.x)
        // -i * b = ( b.y, -b.x)

        state[i].x =  b.y;
        state[i].y = -b.x;

        state[partner].x = -a.y;
        state[partner].y =  a.x;
    }
}
