/// Resolves common gate aliases to their canonical QLang names.
///
/// This allows users to write shorthand versions of gates (e.g., `h`, `x`, `cx`)
/// and have them mapped to their full internal representation (e.g., `hadamard`, `paulix`, `cnot`).
///
/// # Examples
/// ```
/// use qlang::qlang::aliases::resolve_alias;
/// assert_eq!(resolve_alias("h"), "hadamard");
/// assert_eq!(resolve_alias("cx"), "cnot");
/// assert_eq!(resolve_alias("unknown"), "unknown");
/// ```

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

#[cfg(test)]
mod tests {
    use super::resolve_alias;

    #[test]
    fn test_known_aliases() {
        assert_eq!(resolve_alias("h"), "hadamard");
        assert_eq!(resolve_alias("x"), "paulix");
        assert_eq!(resolve_alias("y"), "pauliy");
        assert_eq!(resolve_alias("z"), "pauliz");
        assert_eq!(resolve_alias("cx"), "cnot");
        assert_eq!(resolve_alias("m"), "measure");
        assert_eq!(resolve_alias("d"), "display");
        assert_eq!(resolve_alias("sdg"), "sdagger");
        assert_eq!(resolve_alias("tdg"), "tdagger");
        assert_eq!(resolve_alias("id"), "identity");
    }

    #[test]
    fn test_unknown_alias_returns_itself() {
        assert_eq!(resolve_alias("foo"), "foo");
        assert_eq!(resolve_alias("xyz123"), "xyz123");
    }

    #[test]
    fn test_alias_case_sensitivity() {
        // The current implementation is case-sensitive
        assert_eq!(resolve_alias("H"), "H");
        assert_eq!(resolve_alias("X"), "X");
    }
}
