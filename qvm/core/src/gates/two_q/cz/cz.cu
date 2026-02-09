#include <cuComplex.h>

/// CUDA kernel to apply the Controlled-Z (CZ) gate to a quantum state vector.
///
/// The CZ gate applies a Pauli-Z (i.e., phase flip) to the state
/// if both control and target qubits are in the `|1⟩` state.
///
/// Matrix form (4×4):
/// [ 1  0  0   0 ]
/// [ 0  1  0   0 ]
/// [ 0  0  1   0 ]
/// [ 0  0  0  -1 ]
///
/// # Parameters
/// - `state`: Array of cuDoubleComplex representing the state vector
/// - `control`: Index of control qubit (0-based)
/// - `target`: Index of target qubit (0-based)
/// - `num_qubits`: Total number of qubits in the quantum system
///
/// # Notes
/// - Applies a sign flip to state[i] if bits `control` and `target` are both 1
extern "C" __global__
void cz_kernel(
    cuDoubleComplex* state,
    int control,
    int target,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int control_bit = (i >> control) & 1;
    int target_bit = (i >> target) & 1;

    if (control_bit == 1 && target_bit == 1) {
        state[i].x = -state[i].x;
        state[i].y = -state[i].y;
    }
}
