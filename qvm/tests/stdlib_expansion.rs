use qlang::qlang::QLang;

#[test]
fn test_stdlib_expansion() {
    let mut qlang = QLang::new(3);
    qlang.run_qlang_from_file("examples/stdlib_expansion.ql");
    
    // Check if new functions are registered
    // swap, cz, cy should NOT be here as they are native now
    assert!(!qlang.functions.contains_key("swap"));
    assert!(!qlang.functions.contains_key("cz"));
    assert!(!qlang.functions.contains_key("cy"));
    
    // ch and teleport should be here
    assert!(qlang.functions.contains_key("ch"));
    assert!(qlang.functions.contains_key("teleport"));
    
    // Aliases should be here
    assert!(qlang.functions.contains_key("ccx"));
    assert!(qlang.functions.contains_key("cswap"));
    
    // Check execution
    if qlang.parser.has_errors() {
        println!("Parser errors: {:?}", qlang.parser.get_errors());
        panic!("Parser failed");
    }
    
    assert!(qlang.collapsed);
}

#[test]
fn test_list_functions() {
    let mut qlang = QLang::new(1);
    // We can't easily capture stdout here without a crate, but we can ensure it runs without error.
    let script = r#"
        import std
        list_functions()
    "#;
    qlang.run_from_str(script);
    
    assert!(qlang.functions.contains_key("bell"));
}
