pub fn resolve_alias(raw: &str) -> &str {
    match raw {
        "h" => "hadamard",
        "x" => "paulix",
        "y" => "pauliy",
        "z" => "pauliz",
        "cx" => "cnot",
        "m" => "measure",
        "d" => "display",
        "sdg" => "sdagger",
        "tdg" => "tdagger",
        "id" => "identity",
        _ => raw,
    }
}