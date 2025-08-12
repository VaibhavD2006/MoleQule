"""
QuantumDock: Quantum-Enhanced Drug Discovery Agent
A biologically intelligent quantum docking agent for cisplatin analog discovery.
"""

__version__ = "0.1.0"
__author__ = "QDDA Team"
__email__ = "your-email@example.com"

# Core imports
from .agent_core.data_loader import load_cisplatin_context, load_pancreatic_target
from .agent_core.analog_generator import generate_analogs
from .vqe_engine.vqe_runner import run_vqe_descriptors
from .qnn_model.qnn_predictor import QNNPredictor
from .agent_core.scoring_engine import calculate_final_score, rank_analogs

__all__ = [
    "load_cisplatin_context",
    "load_pancreatic_target", 
    "generate_analogs",
    "run_vqe_descriptors",
    "QNNPredictor",
    "calculate_final_score",
    "rank_analogs"
] 