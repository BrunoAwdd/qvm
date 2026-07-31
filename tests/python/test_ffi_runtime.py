from qlang import QLang, QLangError


def test_python_wrapper_runs_a_gate() -> None:
    runtime = QLang(1)
    runtime.run("x(0)")
    state = runtime.state()
    assert state["num_qubits"] == 1
    assert state["amplitudes"][0]["re"] == 0.0
    assert state["amplitudes"][1]["re"] == 1.0


def test_python_wrapper_does_not_replay_previous_source() -> None:
    runtime = QLang(1)
    runtime.run("x(0)")
    runtime.run("x(0)")
    state = runtime.state()
    assert state["amplitudes"][0]["re"] == 1.0
    assert state["amplitudes"][1]["re"] == 0.0


def test_python_wrapper_surfaces_parser_errors() -> None:
    runtime = QLang(1)
    try:
        runtime.run("h(")
    except QLangError as error:
        assert str(error)
    else:
        raise AssertionError("invalid syntax must raise QLangError")
