#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the RZ(θ) gate to a specific qubit.
///
/// The RZ gate applies opposite phase shifts to `|0⟩` and `|1⟩` basis states:
///
/// ```text
/// RZ(θ) =
/// [ e^{-iθ/2}     0     ]
/// [    0      e^{iθ/2}  ]
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `qubit`: Index of the target qubit (0-based)
/// - `num_qubits`: Total number of qubits in the system
/// - `theta`: Rotation angle in radians
///
/// # Behavior
/// - Multiplies each amplitude by e^{±iθ/2} depending on the qubit value
extern "C" __global__ void rz_kernel(
    cuDoubleComplex* state,
    int qubit,
    int num_qubits,
    double theta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = 1 << num_qubits;
    if (i >= dim) return;

    int bit = (i >> qubit) & 1;
    double angle = (bit == 0) ? -theta / 2.0 : theta / 2.0;

    cuDoubleComplex phase;
    phase.x = cos(angle);
    phase.y = sin(angle);

    cuDoubleComplex v = state[i];
    state[i] = cuCmul(v, phase);
}
