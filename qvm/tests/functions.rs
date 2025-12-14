use qlang::qlang::QLang;

#[test]
fn test_custom_gates_script() {
    let mut qlang = QLang::new(3);
    qlang.run_qlang_from_file("examples/custom_gates.ql");
    
    // Check if functions were registered
    assert!(qlang.functions.contains_key("bell"));
    assert!(qlang.functions.contains_key("teleport_prep"));
    
    // Check if execution finished without error (parser errors would panic in run_qlang_from_file usually, 
    // but let's check parser errors explicitly just in case)
    if qlang.parser.has_errors() {
        println!("Parser errors: {:?}", qlang.parser.get_errors());
        panic!("Parser failed");
    }
    
    // We can also check if measurements were made (collapsed = true)
    assert!(qlang.collapsed);
}

#[test]
fn test_function_scope() {
    let mut qlang = QLang::new(1);
    let script = r#"
        let x = 1
        fn change_x(val) {
            let x = val // Should be local x
        }
        change_x(2)
    "#;
    qlang.run_from_str(script);
    
    // Global x should still be 1
    assert_eq!(*qlang.variables.get("x").unwrap(), 1.0);
}

#[test]
fn test_function_args() {
    let mut qlang = QLang::new(1);
    let script = r#"
        fn my_rx(q, theta) {
            rx(q, theta)
        }
        my_rx(0, 3.14)
    "#;
    qlang.run_from_str(script);
    
    if qlang.parser.has_errors() {
        println!("Parser errors: {:?}", qlang.parser.get_errors());
        panic!("Parser failed");
    }
    
    // Check state (should be rotated by pi ~ |1>)
    let state = qlang.qvm.state_vector();
    assert!((state[1].norm_sqr() - 1.0).abs() < 0.01);
}
