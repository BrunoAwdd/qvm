use qlang::qlang::QLang;

#[test]
fn test_arrays() {
    let mut qlang = QLang::new(3);
    let script = r#"
        let arr = [0, 1, 2]
        h(arr[0])
        cx(arr[0], arr[1])
        cx(arr[1], arr[2])
    "#;
    qlang.run_from_str(script);
    
    // GHZ state: (|000> + |111>) / sqrt(2)
    let state = qlang.qvm.state_vector();
    // |000> is index 0
    // |111> is index 7
    assert!((state[0].norm_sqr() - 0.5).abs() < 1e-6);
    assert!((state[7].norm_sqr() - 0.5).abs() < 1e-6);
}

#[test]
fn test_array_indexing_in_loop() {
    let mut qlang = QLang::new(3);
    let script = r#"
        let qs = [0, 1, 2]
        for (i in 0..3) {
            h(qs[i])
        }
    "#;
    qlang.run_from_str(script);
    
    // All superposed
    let state = qlang.qvm.state_vector();
    for amp in state {
        assert!((amp.norm_sqr() - 0.125).abs() < 1e-6);
    }
}
