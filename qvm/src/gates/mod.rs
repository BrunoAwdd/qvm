pub mod quantum_gate_abstract;
pub mod portal_registry;

pub mod one_q{
    pub mod hadamard;
    pub mod identity;
    pub mod pauli_x;
    pub mod pauli_y;
    pub mod pauli_z;
    pub mod s;
    pub mod s_dagger;
    pub mod t;
    pub mod t_dagger;

}

pub mod general {
    pub mod controlled_u;
}

pub mod rotation_q {
    pub mod phase;
    pub mod rx;
    pub mod ry;
    pub mod rz;
    pub mod u1;
    pub mod u2;
    pub mod u3;
}

pub mod two_q{
    pub mod cnot;
    pub mod cy;
    pub mod cz;
    pub mod iswap;
    pub mod swap;
}

pub mod three_q{
    pub mod toffoli;
    pub mod fredkin;
}














