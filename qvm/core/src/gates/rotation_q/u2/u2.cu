#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the U2(φ, λ) gate to a target qubit.
///
/// The U2 gate is defined as:
/// ```text
/// U2(φ, λ) = 1/√2 × [  1           -e^{iλ}       ]
///                       [  e^{iφ}   e^{i(φ+λ)}    ]
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (cuDoubleComplex[], size 2^n)
/// - `target`: Index of the qubit to apply the gate to
/// - `num_qubits`: Total number of qubits in the state
/// - `phi`: First phase angle (φ)
/// - `lambda`: Second phase angle (λ)
///
/// # Notes
/// - The transformation is only applied to states where the target qubit is 0.
/// - Each such state is paired with the corresponding state where the target qubit is 1.
extern "C"
__global__ void u2_kernel(
    cuDoubleComplex* state,
    int target,
    int num_qubits,
    double phi,
    double lambda
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << num_qubits;
    if (idx >= size) return;

    int mask = 1 << target;

    // Only operate on the |0⟩ component; the |1⟩ partner will be written together
    if ((idx & mask) == 0) {
        int pair = idx | mask;

        cuDoubleComplex a = state[idx];
        cuDoubleComplex b = state[pair];

        double inv_sqrt2 = 1.0 / sqrt(2.0);
        cuDoubleComplex c = make_cuDoubleComplex(inv_sqrt2, 0.0);

        cuDoubleComplex e_il  = make_cuDoubleComplex(cos(lambda), sin(lambda));
        cuDoubleComplex e_ip  = make_cuDoubleComplex(cos(phi), sin(phi));
        cuDoubleComplex e_ipl = make_cuDoubleComplex(cos(phi + lambda), sin(phi + lambda));

        cuDoubleComplex new_a = cuCsub(cuCmul(c, a), cuCmul(cuCmul(c, e_il), b));
        cuDoubleComplex new_b = cuCadd(cuCmul(cuCmul(c, e_ip), a), cuCmul(cuCmul(c, e_ipl), b));

        state[idx]  = new_a;
        state[pair] = new_b;
    }
}
