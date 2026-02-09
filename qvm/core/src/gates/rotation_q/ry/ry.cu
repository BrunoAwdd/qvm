#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the RY(θ) gate to a specific qubit.
///
/// The RY gate rotates the qubit around the Y-axis of the Bloch sphere. It is defined as:
///
/// ```text
/// RY(θ) =
/// [  cos(θ/2)   -sin(θ/2) ]
/// [  sin(θ/2)    cos(θ/2) ]
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `qubit`: Index of the target qubit (0-based)
/// - `num_qubits`: Total number of qubits in the quantum system
/// - `theta`: Rotation angle in radians
///
/// # Notes
/// - This gate uses only real coefficients and does not involve complex phase shifts.
/// - Each amplitude pair differing in the target qubit is rotated together.
extern "C" __global__ void ry_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits,
    double theta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int partner = i ^ (1 << qubit);
    if (i < partner) {
        cuDoubleComplex a = state[i];
        cuDoubleComplex b = state[partner];

        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);

        // RY gate is real-valued — no imaginary parts involved
        state[i].x = c * a.x - s * b.x;
        state[i].y = c * a.y - s * b.y;

        state[partner].x = s * a.x + c * b.x;
        state[partner].y = s * a.y + c * b.y;
    }
}
