// src/qvm/tensor_backend/apply.rs

use ndarray::{Array2, Array4, Array5, s};
use ndarray_linalg::SVD;
use super::TensorBackend;
use num_complex::Complex64;

use crate::{
    gates::quantum_gate_abstract::QuantumGateAbstract, types::qlang_complex::{
        QLangComplex, to_complex64, from_complex64
    }
};

impl TensorBackend {
    pub fn apply_gate(&mut self, gate: &dyn QuantumGateAbstract, qubit: usize) {
        let node = &mut self.network.nodes[qubit];
        let t = &mut node.tensor;

        let mut new_tensor = t.clone();
        let m = gate.matrix();

        for b0 in 0..t.shape()[0] {
            for b1 in 0..t.shape()[2] {
                for i in 0..2 {
                    let mut sum = QLangComplex::default();
                    for j in 0..2 {
                        sum += m[[i, j]] * t[[b0, j, b1]];
                    }
                    new_tensor[[b0, i, b1]] = sum;
                }
            }
        }

        node.tensor = new_tensor;
    }

    pub fn apply_gate_2q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize) {
        let a = &self.network.nodes[q0].tensor;
        let b = &self.network.nodes[q1].tensor;
        let d0 = a.shape()[0];
        let d1 = a.shape()[2]; // = b.shape()[0]
        let d2 = b.shape()[2];

        // 1. Contração A * B => θ
        let mut theta = Array4::from_elem((d0, 2, 2, d2), QLangComplex::default());
        for l in 0..d0 {
            for m in 0..d1 {
                for n in 0..d2 {
                    for i in 0..2 {
                        for j in 0..2 {
                            theta[[l, i, j, n]] += a[[l, i, m]] * b[[m, j, n]];
                        }
                    }
                }
            }
        }

        // 2. Aplicar U (porta 4x4)
        let u = gate.matrix(); // [4,4]
        let mut theta_applied = theta.clone();
        for l in 0..d0 {
            for n in 0..d2 {
                for i in 0..2 {
                    for j in 0..2 {
                        let mut sum = QLangComplex::default();
                        for ip in 0..2 {
                            for jp in 0..2 {
                                let idx_out = (i << 1) | j;
                                let idx_in = (ip << 1) | jp;
                                sum += u[[idx_out, idx_in]] * theta[[l, ip, jp, n]];
                            }
                        }
                        theta_applied[[l, i, j, n]] = sum;
                    }
                }
            }
        }

        // 3. Flatten para 2D: (d0 * 2) x (2 * d2)
        let m: Array2<QLangComplex> = theta_applied
            .map(|c| QLangComplex::new(c.re, c.im))
            .into_shape((d0 * 2, d2 * 2))
            .expect("reshape falhou");

        // 4. SVD
        let m_c64 = to_complex64(&m);
        let (u_opt, s, vt_opt) = m_c64.svd(true, true).expect("falha na SVD");
        let chi = s.len();

        let u_arr = u_opt.expect("U ausente na SVD");
        let vt_arr = vt_opt.expect("Vᵗ ausente na SVD");

        let u_tensor = from_complex64(&u_arr).into_shape((d0, 2, chi)).unwrap();
        let vt_tensor = from_complex64(&vt_arr).into_shape((chi, 2, d2)).unwrap();

        // 5. Converter de volta para QLangComplex e atualizar nodes
        self.network.nodes[q0].tensor = u_tensor.map(|z| QLangComplex::new(z.re, z.im));
        self.network.nodes[q1].tensor = vt_tensor.map(|z| QLangComplex::new(z.re, z.im));

        println!("Aplicado gate 2Q entre q{} e q{} com SVD (χ = {})", q0, q1 + 1, chi);
    }

    pub fn apply_gate_3q(&mut self, gate: &dyn QuantumGateAbstract, q0: usize, q1: usize, q2: usize) {
        let a = &self.network.nodes[q0].tensor;
        let b = &self.network.nodes[q1].tensor;
        let c = &self.network.nodes[q2].tensor;

        let d0 = a.shape()[0];
        let d1 = a.shape()[2]; // = b.shape()[0]
        let d2 = b.shape()[2]; // = c.shape()[0]
        let d3 = c.shape()[2];

        let mut theta = Array5::from_elem((d0, 2, 2, 2, d3), QLangComplex::default());

        for l in 0..d0 {
            for m in 0..d1 {
                for n in 0..d2 {
                    for o in 0..d3 {
                        for i in 0..2 {
                            for j in 0..2 {
                                for k in 0..2 {
                                    theta[[l, i, j, k, o]] += a[[l, i, m]] * b[[m, j, n]] * c[[n, k, o]];
                                }
                            }
                        }
                    }
                }
            }
        }

        let u = gate.matrix(); // [8x8]
        let mut theta_applied = Array5::zeros((d0, 2, 2, 2, d3));

        for l in 0..d0 {
            for o in 0..d3 {
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..2 {
                            let mut sum = QLangComplex::default();
                            for ip in 0..2 {
                                for jp in 0..2 {
                                    for kp in 0..2 {
                                        let idx_out = (i << 2) | (j << 1) | k;
                                        let idx_in = (ip << 2) | (jp << 1) | kp;
                                        sum += u[[idx_out, idx_in]] * theta[[l, ip, jp, kp, o]];
                                    }
                                }
                            }
                            theta_applied[[l, i, j, k, o]] = sum;
                        }
                    }
                }
            }
        }

        // ✅ Agora sim, com theta_applied calculado corretamente
        self.reshape_3q(q0, q1, q2, d0, d3, &theta_applied);
    }


    fn reshape_3q(
        &mut self,
        q0: usize,
        q1: usize,
        q2: usize,
        d0: usize,
        d3: usize,
        theta_applied: &Array5<QLangComplex>,
    ) {
        // 1ª SVD: reshape θ → [d0 * 2, 4 * d3]
        let reshaped_1 = theta_applied
            .clone()
            .into_shape((d0 * 2, 4 * d3))
            .expect("reshape 1 failed");

        let reshaped_1_c64 = to_complex64(&reshaped_1);
        let (u1_opt, s1, vt1_opt) = reshaped_1_c64.svd(true, true).expect("SVD 1 failed");

        let u1 = u1_opt.expect("SVD 1 - U ausente");
        let vt1 = vt1_opt.expect("SVD 1 - VT ausente");
        let chi1 = s1.len(); // valor seguro, consistente com U e VT

        let u1_tensor = from_complex64(&u1)
            .into_shape((d0, 2, chi1))
            .expect("reshape u1_tensor");

        // Truncar VT1 para garantir que reshape funcione
        let expected_cols = 4 * d3; // 2×2×d3
        if vt1.shape()[1] < expected_cols {
            panic!(
                "❌ reshape m2 inválido: vt1.shape = {:?}, esperado ao menos {} colunas (d3 = {})",
                vt1.shape(),
                expected_cols,
                d3
            );
        }
        
        let vt1_trunc = vt1.slice(s![0..chi1, 0..expected_cols]).to_owned();
        let m2 = from_complex64(&vt1_trunc)
            .into_shape((chi1, 2, 2, d3))
            .expect("reshape m2");

        // 2ª SVD: reshape m2 → [chi1 * 2, 2 * d3]
        let reshaped_2 = m2
            .into_shape((chi1 * 2, 2 * d3))
            .expect("reshape 2 failed");

        let reshaped_2_c64 = to_complex64(&reshaped_2);
        let (u2_opt, _s2, vt2_opt) = reshaped_2_c64.svd(true, true).expect("SVD 2 failed");

        let u2 = u2_opt.expect("SVD 2 - U ausente");
        let vt2 = vt2_opt.expect("SVD 2 - VT ausente");
        let shape_u2 = u2.shape(); // geralmente [M, N] onde M = chi1 * 2, N = chi2
        let chi1_inferido = shape_u2[0] / 2;
        let chi2 = shape_u2[1];

        let b_tensor = from_complex64(&u2)
            .into_shape((chi1_inferido, 2, chi2))
            .expect("reshape b_tensor");

        let shape_vt2 = vt2.shape(); // [2, 2]
        let required_len = chi2 * 2 * d3;
        let current_len = vt2.len();
        
        let vt2_padded_c64 = if current_len < required_len {
            let mut padded = Array2::<Complex64>::zeros((chi2, 2 * d3));
            padded.slice_mut(s![..shape_vt2[0], ..shape_vt2[1]]).assign(&vt2);
            padded
        } else {
            vt2.to_owned()
        };

        let c_tensor = from_complex64(&vt2_padded_c64)
            .into_shape((chi2, 2, d3))
            .expect("reshape c_tensor");

        // Atualiza os tensores na rede
        self.network.nodes[q0].tensor = u1_tensor;
        self.network.nodes[q1].tensor = b_tensor;
        self.network.nodes[q2].tensor = c_tensor;

    }
}
