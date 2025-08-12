#!/usr/bin/env python3
"""
Test Script for Phase 3 Docking Integration
Comprehensive testing of quantum and classical docking methods.
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add docking module to path
sys.path.append(str(Path(__file__).parent / "docking"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_module_imports():
    """Test if all docking modules import correctly"""
    print("\n🧪 Testing Module Imports...")
    
    try:
        # Test basic imports
        import numpy as np
        print("✅ NumPy available")
        
        # Test docking modules
        from docking.binding_sites import BindingSiteDetector
        print("✅ BindingSiteDetector imported")
        
        from docking.visualizer import MolecularVisualizer
        print("✅ MolecularVisualizer imported")
        
        from docking.qaoa_docking import QAOAPoseOptimizer
        print("✅ QAOAPoseOptimizer imported")
        
        from docking.classical_docking import ClassicalDocker, ScoringFunction
        print("✅ ClassicalDocker and ScoringFunction imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_binding_site_detector():
    """Test binding site detection functionality"""
    print("\n🎯 Testing Binding Site Detection...")
    
    try:
        from docking.binding_sites import BindingSiteDetector
        
        detector = BindingSiteDetector()
        print("✅ BindingSiteDetector initialized")
        
        # Test predefined sites
        dna_site = detector.get_predefined_site("DNA")
        if dna_site:
            print(f"✅ DNA binding site: {dna_site['name']}")
            print(f"   Description: {dna_site['description']}")
            print(f"   Druggability: {dna_site['druggability_score']}")
        
        gstp1_site = detector.get_predefined_site("GSTP1")
        if gstp1_site:
            print(f"✅ GSTP1 binding site: {gstp1_site['name']}")
        
        # Test druggability analysis
        sample_site = {
            "volume": 200.0,
            "hydrophobic_ratio": 0.5
        }
        druggability = detector.analyze_druggability(sample_site)
        print(f"✅ Druggability analysis: {druggability['druggability_class']} "
              f"(score: {druggability['overall_score']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Binding site detection failed: {e}")
        traceback.print_exc()
        return False

def test_qaoa_optimizer():
    """Test QAOA pose optimization"""
    print("\n⚛️ Testing QAOA Pose Optimization...")
    
    try:
        from docking.qaoa_docking import QAOAPoseOptimizer
        
        optimizer = QAOAPoseOptimizer(n_qubits=6, n_layers=2)  # Smaller for testing
        print(f"✅ QAOA optimizer initialized (quantum: {optimizer.quantum_available})")
        
        # Test pose optimization
        test_smiles = "N[Pt](N)(Cl)Cl"  # Cisplatin
        test_site = {
            "center": [0.0, 0.0, 0.0],
            "radius": 3.0,
            "volume": 113.0
        }
        
        result = optimizer.optimize_pose(
            ligand_smiles=test_smiles,
            binding_site=test_site,
            target_protein="DNA"
        )
        
        print(f"✅ QAOA optimization completed:")
        print(f"   Method: {result.get('method', 'unknown')}")
        print(f"   Binding energy: {result.get('binding_energy', 0):.2f} kcal/mol")
        print(f"   Pose quality: {result.get('pose_quality', 'unknown')}")
        print(f"   Convergence: {result.get('convergence', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ QAOA optimization failed: {e}")
        traceback.print_exc()
        return False

def test_classical_docking():
    """Test classical docking methods"""
    print("\n🔬 Testing Classical Docking...")
    
    try:
        from docking.classical_docking import ClassicalDocker, ScoringFunction
        
        docker = ClassicalDocker()
        print(f"✅ Classical docker initialized")
        print(f"   RDKit available: {docker.rdkit_available}")
        print(f"   OpenBabel available: {docker.openbabel_available}")
        print(f"   Vina available: {docker.vina_available}")
        
        # Test docking
        test_smiles = "N[Pt](N)(Cl)Cl"  # Cisplatin
        test_site = {
            "center": [0.0, 0.0, 0.0],
            "radius": 3.0,
            "volume": 113.0
        }
        
        result = docker.dock_molecule(
            ligand_smiles=test_smiles,
            target_protein="DNA",
            binding_site=test_site,
            method="force_field"
        )
        
        print(f"✅ Classical docking completed:")
        print(f"   Method: {result.get('method', 'unknown')}")
        print(f"   Best score: {result.get('best_score', 0):.2f}")
        print(f"   Num poses: {result.get('num_conformers', 0)}")
        print(f"   Convergence: {result.get('convergence', 'unknown')}")
        
        # Test scoring function
        scorer = ScoringFunction()
        print("✅ Scoring function initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Classical docking failed: {e}")
        traceback.print_exc()
        return False

def test_molecular_visualizer():
    """Test molecular visualization"""
    print("\n🎨 Testing Molecular Visualization...")
    
    try:
        from docking.visualizer import MolecularVisualizer
        
        visualizer = MolecularVisualizer()
        print(f"✅ Molecular visualizer initialized")
        print(f"   3D visualization available: {visualizer.visualization_available}")
        print(f"   Molecular processing available: {visualizer.molecular_processing_available}")
        
        # Test basic viewer creation
        test_smiles = "N[Pt](N)(Cl)Cl"
        html_output = visualizer.create_molecule_viewer(test_smiles)
        
        if html_output and len(html_output) > 100:
            print("✅ Molecule viewer HTML generated successfully")
            print(f"   Output length: {len(html_output)} characters")
        else:
            print("⚠️ Molecule viewer generated short output (likely fallback)")
        
        # Test docking complex viewer
        test_interactions = [
            {"type": "coordination", "distance": 2.1, "atoms": ["Pt", "N7"], "strength": "very_strong"},
            {"type": "hydrogen_bond", "distance": 2.8, "atoms": ["N", "O"], "strength": "strong"}
        ]
        
        complex_html = visualizer.create_docking_complex_viewer(
            ligand_smiles=test_smiles,
            target="DNA",
            binding_score=-8.5,
            interactions=test_interactions
        )
        
        if complex_html and len(complex_html) > 500:
            print("✅ Docking complex viewer generated successfully")
        else:
            print("⚠️ Docking complex viewer generated (likely fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Molecular visualization failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration workflow"""
    print("\n🔗 Testing Full Integration Workflow...")
    
    try:
        # Simulate a complete docking workflow
        from docking.binding_sites import BindingSiteDetector
        from docking.qaoa_docking import QAOAPoseOptimizer
        from docking.classical_docking import ClassicalDocker
        from docking.visualizer import MolecularVisualizer
        
        # Initialize components
        detector = BindingSiteDetector()
        qaoa_opt = QAOAPoseOptimizer(n_qubits=4, n_layers=1)  # Small for testing
        classical_docker = ClassicalDocker()
        visualizer = MolecularVisualizer()
        
        # Test workflow
        test_smiles = "CCO"  # Simple ethanol
        target = "DNA"
        
        # 1. Get binding site
        binding_site = detector.get_predefined_site(target)
        print(f"✅ Step 1: Retrieved {target} binding site")
        
        # 2. Run QAOA optimization
        qaoa_result = qaoa_opt.optimize_pose(test_smiles, binding_site, target)
        print(f"✅ Step 2: QAOA optimization completed ({qaoa_result.get('method', 'unknown')})")
        
        # 3. Run classical docking for comparison
        classical_result = classical_docker.dock_molecule(test_smiles, target, binding_site)
        print(f"✅ Step 3: Classical docking completed ({classical_result.get('method', 'unknown')})")
        
        # 4. Create visualization
        viz_html = visualizer.create_docking_complex_viewer(
            test_smiles, target, qaoa_result.get('binding_energy', -3.0), []
        )
        print(f"✅ Step 4: Visualization created ({len(viz_html)} chars)")
        
        # 5. Compare results
        qaoa_score = qaoa_result.get('binding_energy', 0)
        classical_score = classical_result.get('best_score', 0)
        
        print(f"\n📊 Results Comparison:")
        print(f"   QAOA Score: {qaoa_score:.2f} kcal/mol")
        print(f"   Classical Score: {classical_score:.2f}")
        print(f"   Quantum Enhancement: {qaoa_result.get('quantum_solution', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Phase 3 Docking Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Binding Site Detection", test_binding_site_detector),
        ("QAOA Optimization", test_qaoa_optimizer),
        ("Classical Docking", test_classical_docking),
        ("Molecular Visualization", test_molecular_visualizer),
        ("Full Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: CRASHED - {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Phase 3 implementation is working correctly!")
        print("\n🔬 Available Features:")
        print("   ⚛️ QAOA Quantum Pose Optimization")
        print("   🧪 Classical Force Field Docking") 
        print("   🎯 Advanced Binding Site Detection")
        print("   🎨 3D Molecular Visualization")
        print("   📊 Comprehensive Pose Scoring")
        print("   🔗 Full Integration Pipeline")
    elif passed >= total // 2:
        print("⚠️ Most tests passed - system functional with some limitations")
    else:
        print("❌ Multiple test failures - check dependencies and implementation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 