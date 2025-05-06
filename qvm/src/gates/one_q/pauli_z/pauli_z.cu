#include <cuComplex.h>

/// CUDA kernel to apply the Pauli-Z gate to a specific qubit.
///
/// The Pauli-Z gate flips the phase of the qubit if it's in the `|1⟩` state:
///
/// Z =
/// ```text
/// |  1   0 |
/// |  0  -1 |
/// ```
///
/// This kernel checks the value of the target qubit and flips the sign of the
/// corresponding amplitude in the quantum state vector.
///
/// # Parameters
/// - `state`: Pointer to the quantum state vector (cuDoubleComplex[])
/// - `qubit`: Index of the qubit to apply Pauli-Z to
/// - `num_qubits`: Total number of qubits in the system
///
/// # Note
/// - Only amplitudes where the target qubit is 1 are modified
extern "C" __global__ void pauli_z_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    // If the target qubit is 1, flip the sign
    if ((i >> qubit) & 1) {
        state[i].x = -state[i].x;
        state[i].y = -state[i].y;
    }
}
