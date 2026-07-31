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


def test_python_wrappers_keep_independent_states() -> None:
    with QLang(1) as first, QLang(2) as second:
        first.run("x(0)")

        assert first.num_qubits == 1
        assert second.num_qubits == 2
        assert first.state()["amplitudes"][1]["re"] == 1.0
        assert second.state()["amplitudes"][0]["re"] == 1.0


def test_closed_python_wrapper_rejects_calls() -> None:
    runtime = QLang(1)
    runtime.close()

    try:
        runtime.run("x(0)")
    except QLangError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("a closed runtime must reject calls")
