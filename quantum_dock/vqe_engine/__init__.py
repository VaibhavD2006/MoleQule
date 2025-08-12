"""
VQE Engine Module
Quantum chemistry simulation engine for QuantumDock.
"""

from .vqe_runner import run_vqe_descriptors, apply_context_modifiers

__all__ = [
    "run_vqe_descriptors",
    "apply_context_modifiers"
] 