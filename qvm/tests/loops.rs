use qlang::qlang::QLang;

#[test]
fn test_while_loop() {
    let mut qlang = QLang::new(1);
    let script = r#"
        let i = 0
        while (i < 3) {
            x(0)
            let i = i + 1
        }
    "#;
    qlang.run_from_str(script);
    
    // q0 started at |0>. Flipped 3 times: |1>, |0>, |1>.
    let state = qlang.qvm.state_vector();
    // |1> is index 1.
    assert!((state[1].norm_sqr() - 1.0).abs() < 1e-6);
}

#[test]
fn test_for_loop() {
    let mut qlang = QLang::new(3);
    let script = r#"
        for (i in 0..3) {
            h(i)
        }
    "#;
    qlang.run_from_str(script);
    
    // All qubits in superposition.
    // State |000> + ... + |111>
    // Norm of each amplitude should be 1/sqrt(8) -> prob 1/8
    let state = qlang.qvm.state_vector();
    let prob = 1.0 / 8.0;
    for amp in state {
        assert!((amp.norm_sqr() - prob).abs() < 1e-6);
    }
}
