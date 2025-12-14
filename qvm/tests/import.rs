use qlang::qlang::QLang;

#[test]
fn test_import_std() {
    let mut qlang = QLang::new(2);
    qlang.run_qlang_from_file("examples/import_test.ql");
    
    // Check if functions from stdlib were registered
    assert!(qlang.functions.contains_key("bell"));
    assert!(qlang.functions.contains_key("ghz"));
    
    // Check if execution finished without error
    if qlang.parser.has_errors() {
        println!("Parser errors: {:?}", qlang.parser.get_errors());
        panic!("Parser failed");
    }
    
    // Check if measurements were made
    assert!(qlang.collapsed);
    
    // Check state (Bell state |00> + |11>)
    // Since we measured, it collapsed.
    // But we can check if it ran successfully.
}

#[test]
fn test_import_custom_file() {
    // Create a temporary custom file
    let custom_content = "fn my_func(q) { h(q) }";
    std::fs::write("custom_lib.ql", custom_content).unwrap();
    
    let mut qlang = QLang::new(1);
    let script = r#"
        import custom_lib
        create(1)
        my_func(0)
    "#;
    qlang.run_from_str(script);
    
    assert!(qlang.functions.contains_key("my_func"));
    
    // Cleanup
    std::fs::remove_file("custom_lib.ql").unwrap();
}
