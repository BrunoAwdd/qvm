#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the RX(θ) gate to a single qubit.
///
/// The RX gate rotates the qubit around the X-axis on the Bloch sphere. Its matrix is:
/// ```text
/// RX(θ) = cos(θ/2) * I - i * sin(θ/2) * X
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `qubit`: Target qubit index (0-based)
/// - `num_qubits`: Total number of qubits
/// - `theta`: Rotation angle in radians
///
/// # Note
/// - Each amplitude pair differing in the target qubit is rotated together.
extern "C" __global__ void rx_kernel(
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

        cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0);
        cuDoubleComplex sin_i = cuCmul(make_cuDoubleComplex(s, 0.0), minus_i);  // -i·sin(θ/2)

        // Apply RX(θ)
        state[i]        = cuCadd(cuCmul(make_cuDoubleComplex(c, 0.0), a), cuCmul(sin_i, b));
        state[partner]  = cuCadd(cuCmul(sin_i, a), cuCmul(make_cuDoubleComplex(c, 0.0), b));
    }
}
