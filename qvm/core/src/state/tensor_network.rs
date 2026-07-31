// src/state/tensor_network.rs
use crate::types::qlang_complex::QLangComplex;
use nalgebra::DMatrix;
use ndarray::{s, Array2, Array3};
use num_complex::Complex64;

#[derive(Clone)]
pub struct TensorNode {
    pub tensor: Array3<QLangComplex>,
}

impl TensorNode {
    pub fn identity() -> Self {
        let mut tensor = Array3::from_elem((1, 2, 1), QLangComplex::default());
        tensor[[0, 0, 0]] = QLangComplex::one();
        Self { tensor }
    }
}

/// Representa uma cadeia de tensores conectados para simulação de estado quântico.
#[derive(Clone)]
pub struct TensorNetwork {
    pub nodes: Vec<TensorNode>,
}

impl TensorNetwork {
    pub fn new(num_qubits: usize) -> Self {
        let nodes = (0..num_qubits).map(|_| TensorNode::identity()).collect();
        Self { nodes }
    }

    pub fn to_state_vector(&self) -> Vec<QLangComplex> {
        if self.nodes.is_empty() {
            return vec![QLangComplex::one()];
        }
        let mut partial = Array2::from_elem((1, 1), QLangComplex::one());
        for node in &self.nodes {
            let (basis_count, left_bond) = partial.dim();
            let shape = node.tensor.shape();
            assert_eq!(left_bond, shape[0], "incompatible MPS bond dimensions");
            let mut next = Array2::from_elem((basis_count * 2, shape[2]), QLangComplex::default());
            for basis in 0..basis_count {
                for left in 0..left_bond {
                    for physical in 0..2 {
                        for right in 0..shape[2] {
                            next[[basis * 2 + physical, right]] +=
                                partial[[basis, left]] * node.tensor[[left, physical, right]];
                        }
                    }
                }
            }
            partial = next;
        }
        assert_eq!(partial.shape()[1], 1, "MPS must end with a unit right bond");
        let mps_order: Vec<_> = partial.column(0).iter().copied().collect();
        bit_reverse_order(&mps_order, self.nodes.len())
    }

    pub fn overwrite_from_state_vector(&mut self, state: Vec<QLangComplex>) {
        assert!(
            state.len().is_power_of_two(),
            "state length must be a power of two"
        );
        let n = state.len().trailing_zeros() as usize;
        if n == 0 {
            self.nodes.clear();
            return;
        }
        let reordered = bit_reverse_order(&state, n);
        let mut remainder = Array2::from_shape_vec((1, reordered.len()), reordered).unwrap();
        let mut left_bond = 1;
        self.nodes.clear();

        for site in 0..n - 1 {
            let remaining_qubits = n - site - 1;
            let matrix = remainder
                .into_shape((left_bond * 2, 1 << remaining_qubits))
                .unwrap();
            let (rows, columns) = matrix.dim();
            let values: Vec<_> = matrix
                .iter()
                .map(|value| Complex64::new(value.re, value.im))
                .collect();
            let decomposition = DMatrix::from_row_slice(rows, columns, &values).svd(true, true);
            let u = decomposition.u.unwrap();
            let vt = decomposition.v_t.unwrap();
            let singular = decomposition.singular_values;
            let rank = singular.len();
            let thin_u = Array2::from_shape_fn((rows, rank), |(row, column)| {
                let value = u[(row, column)];
                QLangComplex::new(value.re, value.im)
            });
            let tensor = thin_u.into_shape((left_bond, 2, rank)).unwrap();
            self.nodes.push(TensorNode { tensor });

            let mut weighted_vt = Array2::from_shape_fn((rank, columns), |(row, column)| {
                let value = vt[(row, column)];
                QLangComplex::new(value.re, value.im)
            });
            for row in 0..rank {
                weighted_vt
                    .slice_mut(s![row, ..])
                    .mapv_inplace(|value| value * singular[row]);
            }
            remainder = weighted_vt;
            left_bond = rank;
        }

        let final_tensor = remainder.into_shape((left_bond, 2, 1)).unwrap();
        self.nodes.push(TensorNode {
            tensor: final_tensor,
        });
    }

    pub fn debug_print_network_shapes(&self) {
        println!("🔎 Estado atual da rede de tensores (MPS):");
        for (i, node) in self.nodes.iter().enumerate() {
            let shape = node.tensor.shape();
            println!("  Qubit {} → shape = {:?}", i, shape);
        }
    }
}

fn bit_reverse_order(values: &[QLangComplex], qubits: usize) -> Vec<QLangComplex> {
    let mut reordered = vec![QLangComplex::default(); values.len()];
    for (index, value) in values.iter().enumerate() {
        reordered[reverse_bits(index, qubits)] = *value;
    }
    reordered
}

fn reverse_bits(mut value: usize, width: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..width {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}
