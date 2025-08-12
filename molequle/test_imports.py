#!/usr/bin/env python3
"""
Test script to check if quantum_dock modules can be imported
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

print("Testing quantum_dock imports...")

# Test imports
try:
    from quantum_dock.qnn_model.qnn_predictor import QNNPredictor
    print("✓ QNNPredictor imported successfully")
except ImportError as e:
    print(f"✗ QNNPredictor import failed: {e}")

try:
    from quantum_dock.predictors.admet_predictor import ADMETPredictor
    print("✓ ADMETPredictor imported successfully")
except ImportError as e:
    print(f"✗ ADMETPredictor import failed: {e}")

try:
    from quantum_dock.predictors.synthetic_predictor import SyntheticAccessibilityPredictor
    print("✓ SyntheticAccessibilityPredictor imported successfully")
except ImportError as e:
    print(f"✗ SyntheticAccessibilityPredictor import failed: {e}")

try:
    from quantum_dock.predictors.stability_predictor import StabilityPredictor
    print("✓ StabilityPredictor imported successfully")
except ImportError as e:
    print(f"✗ StabilityPredictor import failed: {e}")

try:
    from quantum_dock.predictors.selectivity_predictor import SelectivityPredictor
    print("✓ SelectivityPredictor imported successfully")
except ImportError as e:
    print(f"✗ SelectivityPredictor import failed: {e}")

try:
    from quantum_dock.predictors.clinical_predictor import ClinicalRelevancePredictor
    print("✓ ClinicalRelevancePredictor imported successfully")
except ImportError as e:
    print(f"✗ ClinicalRelevancePredictor import failed: {e}")

print("\nTesting predictor initialization...")

try:
    qnn = QNNPredictor(n_features=8, n_layers=6, n_qubits=12)
    print("✓ QNNPredictor initialized successfully")
except Exception as e:
    print(f"✗ QNNPredictor initialization failed: {e}")

try:
    admet = ADMETPredictor()
    print("✓ ADMETPredictor initialized successfully")
except Exception as e:
    print(f"✗ ADMETPredictor initialization failed: {e}")

try:
    synthetic = SyntheticAccessibilityPredictor()
    print("✓ SyntheticAccessibilityPredictor initialized successfully")
except Exception as e:
    print(f"✗ SyntheticAccessibilityPredictor initialization failed: {e}")

try:
    stability = StabilityPredictor()
    print("✓ StabilityPredictor initialized successfully")
except Exception as e:
    print(f"✗ StabilityPredictor initialization failed: {e}")

try:
    selectivity = SelectivityPredictor()
    print("✓ SelectivityPredictor initialized successfully")
except Exception as e:
    print(f"✗ SelectivityPredictor initialization failed: {e}")

try:
    clinical = ClinicalRelevancePredictor()
    print("✓ ClinicalRelevancePredictor initialized successfully")
except Exception as e:
    print(f"✗ ClinicalRelevancePredictor initialization failed: {e}")

print("\nAll tests completed!") 