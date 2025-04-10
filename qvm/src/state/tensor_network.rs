// src/state/tensor_network.rs
use ndarray::{Array1, Array2, Array3, Ix2};
use ndarray_linalg::SVD;
use crate::types::qlang_complex::{
        QLangComplex, to_complex64, from_complex64
    };
use crate::qvm::tensor_backend::contract::contract;


pub struct TensorNode {
    pub tensor: Array3<QLangComplex>,
}

impl TensorNode {
    pub fn identity() -> Self {
        let mut tensor = Array3::from_elem((1, 2, 1), QLangComplex::default());
        tensor[[0, 0, 0]] = QLangComplex::new(1.0, 0.0); 
        Self { tensor }
    }
}

/// Representa uma cadeia de tensores conectados para simulação de estado quântico.
pub struct TensorNetwork {
    pub nodes: Vec<TensorNode>,
}

impl TensorNetwork {
    pub fn new(num_qubits: usize) -> Self {
        let nodes = (0..num_qubits)
            .map(|_| TensorNode::identity())
            .collect();
        Self { nodes }
    }

    pub fn to_state_vector(&self) -> Vec<QLangComplex> {
        //self.debug_print_network_shapes();

        let mut state: Option<Array2<QLangComplex>> = None;

        for (i, node) in self.nodes.iter().enumerate() {
            let t = &node.tensor;
            let shape = t.shape(); // [D0, 2, D1]

            let reshaped = if i == 0 {
                t.clone()
                    .into_shape((shape[1], shape[2])) // (2, D)
                    .expect("reshape do primeiro tensor falhou")
            } else if i == self.nodes.len() - 1 {
                // Último tensor
                t.clone()
                    .into_shape((shape[0] * shape[1], shape[2])) // (D×2, D)
                    .expect("reshape do último tensor falhou")
            } else {
                t.clone()
                    .into_shape((shape[0], shape[1] * shape[2])) // (D, 2×D)
                    .expect("reshape do tensor falhou")
            };

            if let Some(current_state) = state {
                if current_state.shape()[1] != reshaped.shape()[0] {
                    panic!(
                        "❌ Erro de contração {}: state = {:?}, tensor = {:?} — dim compartilhada não bate: {} != {}",
                        i,
                        current_state.dim(),
                        reshaped.dim(),
                        current_state.shape()[1],
                        reshaped.shape()[0]
                    );
                }

                state = Some(contract(current_state, reshaped));
            } else {
                state = Some(reshaped);
            }
        }

        state.expect("rede vazia").iter().cloned().collect()
    }

    pub fn overwrite_from_state_vector(&mut self, state: Vec<QLangComplex>) {
        let n = (state.len() as f64).log2().round() as usize;
        self.nodes = Vec::with_capacity(n);

        let mut reshaped = Array1::from(state.clone()).into_shape((1, state.len())).unwrap();
        for _ in 0..n {
            let dim = reshaped.shape()[1];
            let reshaped_c64 = to_complex64(&reshaped.into_dimensionality::<Ix2>().unwrap());

            // reshape para [2, rest] e aplicar SVD
            let reshaped2 = reshaped_c64.into_shape((2, dim / 2)).unwrap();
            let (u_opt, _, v_opt) = reshaped2.svd(true, true).unwrap();
            let u = u_opt.unwrap();
            let v = v_opt.unwrap();

            // cortar em [1, 2, D1]
            let tensor = from_complex64(&u)
                .into_shape((1, 2, u.shape()[1]))
                .unwrap();

            self.nodes.push(TensorNode { tensor });

            reshaped = from_complex64(&v)
                .into_shape((1, v.shape()[0] * v.shape()[1]))
                .unwrap();
        }
        let reshaped = reshaped; // garante que é o mesmo tipo

        println!("🔎 Último reshape: reshaped.len() = {}", reshaped.len());
        // Último tensor: se sobrar apenas um escalar, finalize
        if reshaped.len() == 1 {
            let mut final_tensor = Array3::zeros((1, 2, 1));
            final_tensor[[0, 0, 0]] = reshaped.iter().next().cloned().unwrap();
            self.nodes.push(TensorNode { tensor: final_tensor });
        } else {
            assert_eq!(reshaped.len() % 2, 0, "❌ Último reshape inválido");
            let final_tensor = reshaped.clone()
                .into_shape((reshaped.len() / 2, 2, 1))
                .expect("❌ Último reshape do estado falhou");
            self.nodes.push(TensorNode { tensor: final_tensor });
        }
    }


    pub fn debug_print_network_shapes(&self) {
        println!("🔎 Estado atual da rede de tensores (MPS):");
        for (i, node) in self.nodes.iter().enumerate() {
            let shape = node.tensor.shape();
            println!("  Qubit {} → shape = {:?}", i, shape);
        }
}

}



