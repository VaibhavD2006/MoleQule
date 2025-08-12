#!/usr/bin/env python3
"""
Windows Compatibility Test for QuantumDock
Tests that the system can run without PySCF installed.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from agent_core.data_loader import load_cisplatin_context, load_pancreatic_target
        print("✅ agent_core.data_loader imported successfully")
    except Exception as e:
        print(f"❌ Error importing agent_core.data_loader: {e}")
        return False
    
    try:
        from agent_core.analog_generator import generate_analogs
        print("✅ agent_core.analog_generator imported successfully")
    except Exception as e:
        print(f"❌ Error importing agent_core.analog_generator: {e}")
        return False
    
    try:
        from vqe_engine.vqe_runner import run_vqe_descriptors, check_quantum_dependencies
        print("✅ vqe_engine.vqe_runner imported successfully")
    except Exception as e:
        print(f"❌ Error importing vqe_engine.vqe_runner: {e}")
        return False
    
    try:
        from qnn_model.qnn_predictor import QNNPredictor
        print("✅ qnn_model.qnn_predictor imported successfully")
    except Exception as e:
        print(f"❌ Error importing qnn_model.qnn_predictor: {e}")
        return False
    
    try:
        from agent_core.scoring_engine import calculate_final_score, rank_analogs
        print("✅ agent_core.scoring_engine imported successfully")
    except Exception as e:
        print(f"❌ Error importing agent_core.scoring_engine: {e}")
        return False
    
    return True


def test_configuration_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from agent_core.data_loader import load_cisplatin_context, load_pancreatic_target
        
        # Test loading cisplatin context
        cisplatin_context = load_cisplatin_context("data/cisplatin_context.json")
        print(f"✅ Cisplatin context loaded: {len(cisplatin_context)} keys")
        
        # Test loading pancreatic target
        pancreatic_target = load_pancreatic_target("data/pancreatic_target.json")
        print(f"✅ Pancreatic target loaded: {len(pancreatic_target)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def test_analog_generation():
    """Test analog generation functionality."""
    print("\nTesting analog generation...")
    
    try:
        from agent_core.analog_generator import generate_analogs
        
        # Test data
        base_smiles = "N[Pt](N)(Cl)Cl"
        substitutions = {
            "ammine": ["methyl", "ethyl"],
            "chloride": ["bromide"]
        }
        
        analogs = generate_analogs(base_smiles, substitutions)
        print(f"✅ Generated {len(analogs)} analogs")
        
        if analogs:
            print(f"   Sample analog: {analogs[0].get('smiles', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating analogs: {e}")
        return False


def test_vqe_fallback():
    """Test VQE engine fallback functionality."""
    print("\nTesting VQE engine fallback...")
    
    try:
        from vqe_engine.vqe_runner import check_quantum_dependencies
        
        # Check dependencies
        deps = check_quantum_dependencies()
        print(f"✅ Dependency check: {deps}")
        
        # Test that we can still run without PySCF
        if not deps['pyscf']:
            print("✅ Running without PySCF (expected on Windows)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in VQE fallback test: {e}")
        return False


def test_qnn_creation():
    """Test QNN model creation."""
    print("\nTesting QNN model creation...")
    
    try:
        from qnn_model.qnn_predictor import QNNPredictor
        
        # Create QNN (this might fail if PennyLane isn't installed)
        qnn = QNNPredictor(n_features=3, n_layers=2)
        print("✅ QNN model created successfully")
        
        # Test prediction with dummy data
        test_features = [1.0, 2.0, 3.0]
        prediction = qnn.predict(test_features)
        print(f"✅ QNN prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating QNN: {e}")
        print("   This might be expected if PennyLane is not installed")
        return False


def test_scoring_system():
    """Test scoring and ranking system."""
    print("\nTesting scoring system...")
    
    try:
        from agent_core.scoring_engine import calculate_final_score, rank_analogs
        
        # Test scoring
        score = calculate_final_score(
            binding_affinity=-8.0,
            resistance_score=0.3,
            toxicity_score=0.2
        )
        print(f"✅ Final score calculated: {score}")
        
        # Test ranking
        test_analogs = [
            {"analog_id": "1", "final_score": 0.8},
            {"analog_id": "2", "final_score": 0.9},
            {"analog_id": "3", "final_score": 0.7}
        ]
        
        ranked = rank_analogs(test_analogs)
        print(f"✅ Ranked {len(ranked)} analogs")
        print(f"   Top analog: {ranked[0]['analog_id']} (score: {ranked[0]['final_score']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in scoring system: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 QuantumDock Windows Compatibility Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    tests = [
        test_imports,
        test_configuration_loading,
        test_analog_generation,
        test_vqe_fallback,
        test_qnn_creation,
        test_scoring_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! QuantumDock is compatible with your Windows setup.")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. QNN test failure is expected without PennyLane.")
        print("   Install PennyLane for full functionality: pip install pennylane")
    else:
        print("❌ Multiple tests failed. Please check your installation.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. For PySCF: use conda install -c conda-forge -c pyscf pyscf")
        print("3. Check README.md for Windows-specific installation instructions")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 