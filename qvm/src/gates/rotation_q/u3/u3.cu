#include <cuComplex.h>

extern "C" __global__
void u3_kernel(cuDoubleComplex* state, int target, int n, double theta, double phi, double lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = 1 << n;

    if (idx >= size) return;

    int mask = 1 << target;

    // Só processa pares onde o bit do qubit alvo está 0
    if ((idx & mask) == 0) {
        int pair = idx | mask;

        cuDoubleComplex a = state[idx];
        cuDoubleComplex b = state[pair];

        double ct = cos(theta / 2.0);
        double st = sin(theta / 2.0);

        cuDoubleComplex e_il = make_cuDoubleComplex(cos(lambda), sin(lambda));
        cuDoubleComplex e_ip = make_cuDoubleComplex(cos(phi), sin(phi));
        cuDoubleComplex e_ipl = make_cuDoubleComplex(cos(phi + lambda), sin(phi + lambda));

        cuDoubleComplex new_a = cuCsub(cuCmul(make_cuDoubleComplex(ct, 0.0), a),
                                       cuCmul(cuCmul(e_il, make_cuDoubleComplex(st, 0.0)), b));

        cuDoubleComplex new_b = cuCadd(cuCmul(cuCmul(e_ip, make_cuDoubleComplex(st, 0.0)), a),
                                       cuCmul(cuCmul(e_ipl, make_cuDoubleComplex(ct, 0.0)), b));

        state[idx] = new_a;
        state[pair] = new_b;
    }
}
