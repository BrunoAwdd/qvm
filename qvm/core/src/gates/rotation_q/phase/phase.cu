#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply a generic phase gate with angle θ to a specific qubit.
///
/// The gate performs the following unitary transformation:
/// ```text
/// | 1       0 |
/// | 0   e^{iθ} |
/// ```
///
/// This generalizes the S (θ=π/2), T (θ=π/4), and TDagger (θ=−π/4) gates.
///
/// # Parameters
/// - `state`: Quantum state vector (`cuDoubleComplex*`, size 2^n)
/// - `qubit`: Index of the target qubit (0-based)
/// - `num_qubits`: Total number of qubits in the system
/// - `theta`: Phase angle in radians (θ)
///
/// # Behavior
/// - Multiplies amplitudes where the target qubit is `1` by `e^{iθ}`.
extern "C" __global__ void phase_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits,
    double theta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        cuDoubleComplex phase;
        phase.x = cos(theta);  // Real part: cos(θ)
        phase.y = sin(theta);  // Imag part: sin(θ)

        cuDoubleComplex v = state[i];
        state[i] = cuCmul(v, phase); // Multiply by e^{iθ}
    }
}
