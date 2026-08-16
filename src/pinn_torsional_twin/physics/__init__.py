"""Governing physics and residual operators."""

from .two_inertia import simulate_two_inertia, sigmoid_stiffness_reference
from .weak import (
    build_constant_weak_terms,
    build_sigmoid_weak_terms,
    weak_sigmoid_losses,
)

__all__ = [
    "build_constant_weak_terms",
    "build_sigmoid_weak_terms",
    "simulate_two_inertia",
    "sigmoid_stiffness_reference",
    "weak_sigmoid_losses",
]

