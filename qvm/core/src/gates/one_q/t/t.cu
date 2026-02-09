#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the T gate (π/8 gate) to a specific qubit.
///
/// The T gate applies a π/4 phase (i.e. multiplies `|1⟩` by `e^{iπ/4}`).
/// It is defined as:
///
/// ```text
/// | 1          0 |
/// | 0   e^{iπ/4} |
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (cuDoubleComplex[], length 2^n)
/// - `qubit`: The target qubit index (0-based)
/// - `num_qubits`: Total number of qubits in the quantum register
///
/// # Note
/// - Only the amplitudes where the target qubit is `1` are phase-shifted.
extern "C" __global__ void t_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    if ((i >> qubit) & 1) {
        double angle = M_PI / 4.0;
        cuDoubleComplex phase = make_cuDoubleComplex(cos(angle), sin(angle)); // e^{iπ/4}
        state[i] = cuCmul(state[i], phase);
    }
}
