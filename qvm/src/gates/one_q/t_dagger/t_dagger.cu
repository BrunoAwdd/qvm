#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the T† (T-dagger) gate to a specific qubit.
///
/// This gate applies a negative π/4 phase shift to the `|1⟩` state:
///
/// ```text
/// | 1           0 |
/// | 0   e^{-iπ/4} |
/// ```
///
/// It's the inverse of the T gate (π/8 gate).
///
/// # Parameters
/// - `state`: Quantum state vector (`cuDoubleComplex*`, size 2^n)
/// - `target`: The index of the qubit to apply the T† gate to
/// - `num_qubits`: Total number of qubits in the system
///
/// # Behavior
/// - Multiplies amplitudes where the target qubit is `1` by e^{-iπ/4}
extern "C" __global__ void tdagger_kernel(
    cuDoubleComplex* state,
    int target,
    int num_qubits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;
    if (idx >= size) return;

    if ((idx >> target) & 1) {
        double angle = -M_PI / 4.0;
        cuDoubleComplex phase = make_cuDoubleComplex(cos(angle), sin(angle)); // e^{-iπ/4}
        state[idx] = cuCmul(state[idx], phase);
    }
}
