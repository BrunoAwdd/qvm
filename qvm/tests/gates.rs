mod gates {
    mod one_q {
        mod identity_test;
        mod pauli_x_test;
        mod pauli_y_test;
        mod pauli_z_test;
        mod rx_test;
        mod ry_test;
        mod rz_test;
        mod s_test;
        mod s_dagger_test;
        mod t_test;
        mod t_dagger_test;
        mod u1_test;
        mod u2_test;
        mod u3_test;
    }
    mod two_q {
        mod cnot_test;
        mod swap_test;
    }
    mod three_q {
        mod toffoli_test;
        mod fredkin_test;
    }   
}