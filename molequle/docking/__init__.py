"""
MoleQule Docking Service Package
Advanced molecular docking with quantum and classical methods.
"""

__version__ = "2.0.0"
__author__ = "MoleQule Team"

# Export main components
try:
    from .main import app
    from .binding_sites import BindingSiteDetector
    from .visualizer import MolecularVisualizer
    from .qaoa_docking import QAOAPoseOptimizer
    from .classical_docking import ClassicalDocker, ScoringFunction
    
    __all__ = [
        'app',
        'BindingSiteDetector',
        'MolecularVisualizer', 
        'QAOAPoseOptimizer',
        'ClassicalDocker',
        'ScoringFunction'
    ]
    
except ImportError:
    # Fallback when dependencies are not available
    __all__ = [] 