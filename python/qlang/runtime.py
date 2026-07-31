from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any


class QLangError(RuntimeError):
    """Raised when the native QLang runtime rejects an operation."""


def _library_candidates() -> list[Path]:
    override = os.environ.get("QLANG_LIBRARY")
    names = ("libqlang_ffi.so", "libqlang_ffi.dylib", "qlang_ffi.dll")
    candidates = [Path(override)] if override else []
    project = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidates.extend(project / "qvm" / "target" / profile / name for name in names)
    return candidates


def _load_library() -> ctypes.CDLL:
    for candidate in _library_candidates():
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    searched = "\n".join(str(path) for path in _library_candidates())
    raise QLangError(
        "QLang native library was not found. Build it with "
        "`cargo build -p qlang-ffi --release` or set QLANG_LIBRARY.\n"
        f"Searched:\n{searched}"
    )


class QLang:
    """Small, memory-safe ctypes wrapper around the qlang-ffi ABI."""

    def __init__(self, num_qubits: int, library: str | os.PathLike[str] | None = None):
        self._lib = ctypes.CDLL(str(library)) if library else _load_library()
        self._configure_abi()
        self._handle = self._lib.qlang_create(num_qubits)
        if not self._handle:
            self._check(1)

    def _configure_abi(self) -> None:
        self._lib.qlang_create.argtypes = [ctypes.c_size_t]
        self._lib.qlang_create.restype = ctypes.c_void_p
        self._lib.qlang_destroy.argtypes = [ctypes.c_void_p]
        self._lib.qlang_destroy.restype = None
        self._lib.qlang_run_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.qlang_run_source.restype = ctypes.c_int
        self._lib.qlang_reset.argtypes = [ctypes.c_void_p]
        self._lib.qlang_reset.restype = ctypes.c_int
        self._lib.qlang_num_qubits.argtypes = [ctypes.c_void_p]
        self._lib.qlang_num_qubits.restype = ctypes.c_size_t
        self._lib.qlang_measure_all.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self._lib.qlang_measure_all.restype = ctypes.c_ssize_t
        self._lib.qlang_state_json.argtypes = [ctypes.c_void_p]
        self._lib.qlang_state_json.restype = ctypes.c_void_p
        self._lib.qlang_last_error.argtypes = []
        self._lib.qlang_last_error.restype = ctypes.c_char_p
        self._lib.qlang_string_free.argtypes = [ctypes.c_void_p]
        self._lib.qlang_string_free.restype = None

    def _check(self, status: int) -> None:
        if status == 0:
            return
        raw = self._lib.qlang_last_error()
        message = raw.decode("utf-8") if raw else f"native error status {status}"
        raise QLangError(message)

    def _require_handle(self) -> ctypes.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise QLangError("QLang runtime is closed")
        return handle

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.qlang_destroy(handle)
            self._handle = None

    def __enter__(self) -> QLang:
        self._require_handle()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown, after ctypes
            # globals have already started being torn down.
            pass

    @property
    def num_qubits(self) -> int:
        return int(self._lib.qlang_num_qubits(self._require_handle()))

    def run(self, source: str) -> None:
        self._check(
            self._lib.qlang_run_source(
                self._require_handle(), source.encode("utf-8")
            )
        )

    def reset(self) -> None:
        self._check(self._lib.qlang_reset(self._require_handle()))

    def measure_all(self) -> list[int]:
        output = (ctypes.c_uint8 * self.num_qubits)()
        written = self._lib.qlang_measure_all(self._require_handle(), output, len(output))
        if written < 0:
            self._check(1)
        return list(output[:written])

    def state(self) -> dict[str, Any]:
        pointer = self._lib.qlang_state_json(self._require_handle())
        if not pointer:
            self._check(1)
        try:
            payload = ctypes.cast(pointer, ctypes.c_char_p).value
            if payload is None:
                raise QLangError("native runtime returned an empty state")
            return json.loads(payload.decode("utf-8"))
        finally:
            self._lib.qlang_string_free(pointer)
