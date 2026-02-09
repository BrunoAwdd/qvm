#include <cuComplex.h>
#include <math.h>

/// CUDA kernel to apply the U3(θ, φ, λ) gate to a single qubit.
///
/// The U3 gate is a universal single-qubit gate defined as:
/// ```text
/// U3(θ, φ, λ) =
/// [  cos(θ/2)              -e^{iλ}·sin(θ/2) ]
/// [  e^{iφ}·sin(θ/2)   e^{i(φ+λ)}·cos(θ/2) ]
/// ```
///
/// # Parameters
/// - `state`: Quantum state vector (array of cuDoubleComplex)
/// - `target`: Index of the target qubit
/// - `n`: Total number of qubits
/// - `theta`, `phi`, `lambda`: Parameters of the U3 gate in radians
///
/// # Behavior
/// - Each thread processes a pair of amplitudes (|x0⟩ and |x1⟩).
/// - Threads only process the `|0⟩` side of the pair to avoid race conditions.
extern "C" __global__
void u3_kernel(
    cuDoubleComplex* state,
    int target,
    int n,
    double theta,
    double phi,
    double lambda
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << n;

    if (idx >= size) return;

    int mask = 1 << target;

    // Only process one side of the pair
    if ((idx & mask) == 0) {
        int pair = idx | mask;

        cuDoubleComplex a = state[idx];
        cuDoubleComplex b = state[pair];

        double ct = cos(theta / 2.0);
        double st = sin(theta / 2.0);

        cuDoubleComplex e_il   = make_cuDoubleComplex(cos(lambda), sin(lambda));
        cuDoubleComplex e_ip   = make_cuDoubleComplex(cos(phi), sin(phi));
        cuDoubleComplex e_ipl  = make_cuDoubleComplex(cos(phi + lambda), sin(phi + lambda));

        cuDoubleComplex new_a = cuCsub(
            cuCmul(make_cuDoubleComplex(ct, 0.0), a),
            cuCmul(cuCmul(e_il, make_cuDoubleComplex(st, 0.0)), b)
        );

        cuDoubleComplex new_b = cuCadd(
            cuCmul(cuCmul(e_ip, make_cuDoubleComplex(st, 0.0)), a),
            cuCmul(cuCmul(e_ipl, make_cuDoubleComplex(ct, 0.0)), b)
        );

        state[idx] = new_a;
        state[pair] = new_b;
    }
}
