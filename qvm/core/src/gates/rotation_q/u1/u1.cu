#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the U1(λ) gate to a target qubit.
///
/// The U1 gate is a phase gate defined by:
///
/// ```text
/// U1(λ) =
/// [ 1       0     ]
/// [ 0   e^{iλ}    ]
/// ```
///
/// It multiplies the `|1⟩` component of the target qubit by `e^{iλ}`.
///
/// # Parameters
/// - `state`: The quantum state vector (array of cuDoubleComplex)
/// - `target`: Index of the qubit to apply U1 to (0-based)
/// - `n`: Total number of qubits
/// - `lambda`: Phase angle λ in radians
///
/// # Notes
/// - Amplitudes where the target qubit is `1` are multiplied by e^{iλ}
extern "C" __global__
void u1_kernel(cuDoubleComplex* state, int target, int n, double lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << n;
    if (idx >= size) return;

    int mask = 1 << target;
    if (idx & mask) {
        cuDoubleComplex e_il = make_cuDoubleComplex(cos(lambda), sin(lambda)); // e^{iλ}
        state[idx] = cuCmul(e_il, state[idx]);
    }
}
