use ndarray::Array2;
use crate::types::qlang_complex::QLangComplex;
use crate::gates::quantum_gate_abstract::QuantumGateAbstract;

/// A two-qubit controlled gate for applying an arbitrary single-qubit unitary
/// when the control qubit is `1`.
///
/// # Matrix Representation
/// This gate creates a 4x4 block matrix where:
/// - The upper-left 2x2 block is the identity (no operation when control is `0`)
/// - The lower-right 2x2 block is the provided unitary `U`
///
/// # Example
/// ```
/// let cu = ControlledU::new_real(0.0, 1.0, 1.0, 0.0); // acts like a controlled-X (CNOT)
/// ```
pub struct ControlledU {
    /// The full 4x4 matrix representing the controlled-U gate
    matrix: Array2<QLangComplex>,
}

impl QuantumGateAbstract for ControlledU {
    /// Returns the full 4x4 matrix representation of the gate.
    fn matrix(&self) -> Array2<QLangComplex> {
        self.matrix.clone()
    }

    /// Returns the string identifier for this gate.
    fn name(&self) -> &'static str {
        "cu"
    }
}

impl ControlledU {
    /// Creates a new `ControlledU` gate from a 2x2 unitary matrix.
    ///
    /// # Parameters
    /// - `u`: A 2x2 unitary matrix of complex values (`QLangComplex`)
    /// - `name`: Optional name or label (not currently used)
    ///
    /// # Panics
    /// Panics if the provided matrix is not 2x2.
    pub fn new(u: Array2<QLangComplex>, _name: Option<String>) -> Self {
        assert_eq!(u.dim(), (2, 2), "ControlledU expects a 2x2 unitary matrix");

        let zero = QLangComplex::zero();
        let one = QLangComplex::one();

        let mut matrix = Array2::from_elem((4, 4), zero);
        matrix[[0, 0]] = one;
        matrix[[1, 1]] = one;

        for i in 0..2 {
            for j in 0..2 {
                matrix[[i + 2, j + 2]] = u[[i, j]];
            }
        }

        Self { matrix }
    }

    /// Convenience constructor for creating a real-valued controlled-U gate.
    ///
    /// # Parameters
    /// - `u00`, `u01`, `u10`, `u11`: Elements of a 2x2 real matrix
    ///
    /// # Returns
    /// A new `ControlledU` instance with the corresponding real-valued unitary.
    pub fn new_real(u00: f64, u01: f64, u10: f64, u11: f64) -> Self {
        use ndarray::array;
        let u = array![
            [QLangComplex::new(u00, 0.0), QLangComplex::new(u01, 0.0)],
            [QLangComplex::new(u10, 0.0), QLangComplex::new(u11, 0.0)],
        ];
        Self::new(u, None)
    }
}

#[cfg(test)]
mod tests {
    mod cpu {
        use super::super::*;
        use ndarray::array;

        #[test]
        fn test_controlled_u_from_real() {
            // Define a NOT (Pauli-X) matrix
            let cu = ControlledU::new_real(0.0, 1.0, 1.0, 0.0);
            let matrix = cu.matrix();

            // Identity on control=0
            assert_eq!(matrix[[0, 0]], QLangComplex::one());
            assert_eq!(matrix[[1, 1]], QLangComplex::one());
            assert_eq!(matrix[[0, 1]], QLangComplex::zero());
            assert_eq!(matrix[[1, 0]], QLangComplex::zero());

            // X gate on control=1
            assert_eq!(matrix[[2, 3]], QLangComplex::one());
            assert_eq!(matrix[[3, 2]], QLangComplex::one());
            assert_eq!(matrix[[2, 2]], QLangComplex::zero());
            assert_eq!(matrix[[3, 3]], QLangComplex::zero());
        }

        #[test]
        fn test_controlled_u_from_matrix() {
            let u = array![
                [QLangComplex::new(1.0, 0.0), QLangComplex::new(0.0, 0.0)],
                [QLangComplex::new(0.0, 0.0), QLangComplex::new(-1.0, 0.0)],
            ];
            let cu = ControlledU::new(u.clone(), None);
            let matrix = cu.matrix();

            // Identity on upper block
            assert_eq!(matrix[[0, 0]], QLangComplex::one());
            assert_eq!(matrix[[1, 1]], QLangComplex::one());

            // Inserted unitary on bottom-right block
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(matrix[[i + 2, j + 2]], u[[i, j]]);
                }
            }
        }

        #[test]
        fn test_controlled_u_name() {
            let cu = ControlledU::new_real(1.0, 0.0, 0.0, 1.0);
            assert_eq!(cu.name(), "cu");
        }
    }

}

