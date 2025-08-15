"""
Numba setup and conditional imports.
"""

import os
from typing import Callable

# Debug flag setup
DEBUG_MODE = os.environ.get("ADABOOST_DEBUG", "False").lower() in ("true", "1", "yes")


def setup_numba():
    """
    Setup Numba imports based on debug mode.

    Returns:
        tuple: (njit decorator, prange function)
    """
    if DEBUG_MODE:
        print("🐛 DEBUG MODE - Numba disabled")

        def njit(*args, **kwargs) -> Callable:
            """Dummy njit decorator for debugging."""

            def decorator(func: Callable) -> Callable:
                return func

            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        def prange(n: int):
            """Dummy prange for debugging."""
            return range(n)

        return njit, prange
    else:
        print("🚀 PRODUCTION MODE - Numba enabled")
        from numba import njit, prange

        return njit, prange


# Initialize numba functions
njit, prange = setup_numba()
