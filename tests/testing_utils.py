from typing import Callable, Any


def get_func_variants(func: Callable[..., Any]) -> list[tuple[str, Callable[..., Any]]]:
    """Returns a list of (name, function) for both JIT compiled and pure python versions.

    This is used to parametrize pytest test cases to ensure 100% coverage and
    behavioral parity between the Numba LLVM bytecode and native Python execution.
    """
    return [
        (f"{func.__name__}_jit", func),
        (f"{func.__name__}_py", getattr(func, "py_func", func)),
    ]
