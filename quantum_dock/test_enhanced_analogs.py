#!/usr/bin/env python3
"""
Test script for Enhanced Analog Generator
Verifies that 30 pharmaceutical-grade analogs are generated correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_enhanced_analog_generator():
    """Test the enhanced analog generator."""
    print("🧪 Testing Enhanced Analog Generator...")
    print("=" * 50)
    
    try:
        from agent_core.enhanced_analog_generator import generate_enhanced_analogs_30
        
        # Generate analogs
        print("🔬 Generating 30 enhanced analogs...")
        analogs = generate_enhanced_analogs_30("N[Pt](N)(Cl)Cl")
        
        print(f"✅ Generated {len(analogs)} analogs")
        print()
        
        # Test results
        if len(analogs) == 0:
            print("❌ ERROR: No analogs generated!")
            return False
        
        print("📊 Analog Summary:")
        print(f"   Total analogs: {len(analogs)}")
        
        # Analyze analog types
        type_counts = {}
        for analog in analogs:
            analog_type = analog.get('analog_type', 'unknown')
            type_counts[analog_type] = type_counts.get(analog_type, 0) + 1
        
        for analog_type, count in type_counts.items():
            print(f"   {analog_type}: {count} analogs")
        
        print()
        print("🔍 Sample Analogs:")
        
        # Show first 5 analogs
        for i, analog in enumerate(analogs[:5]):
            print(f"   {i+1}. ID: {analog['id'][:8]}...")
            print(f"      SMILES: {analog['smiles']}")
            print(f"      Type: {analog['analog_type']}")
            print(f"      Substitution: {analog['substitution']}")
            
            if 'molecular_weight' in analog:
                print(f"      MW: {analog['molecular_weight']:.1f} Da")
            if 'druglike_score' in analog:
                print(f"      Drug-like Score: {analog['druglike_score']:.3f}")
            if 'logP' in analog:
                print(f"      LogP: {analog['logP']:.2f}")
            print()
        
        # Validate drug-likeness
        druglike_analogs = [a for a in analogs if a.get('druglike_score', 0) > 0.5]
        print(f"💊 Drug-like analogs (score > 0.5): {len(druglike_analogs)}/{len(analogs)}")
        
        # Validate molecular weights
        mw_analogs = [a for a in analogs if 200 <= a.get('molecular_weight', 0) <= 700]
        print(f"⚖️  Optimal MW analogs (200-700 Da): {len(mw_analogs)}/{len(analogs)}")
        
        # Check for diversity
        unique_types = len(set(a.get('analog_type', 'unknown') for a in analogs))
        print(f"🌈 Analog type diversity: {unique_types} different types")
        
        print()
        print("🎯 Test Results:")
        
        # Success criteria
        success_criteria = [
            (len(analogs) >= 20, f"Generated at least 20 analogs: {len(analogs)}"),
            (len(druglike_analogs) >= 15, f"At least 15 drug-like analogs: {len(druglike_analogs)}"),
            (unique_types >= 3, f"At least 3 analog types: {unique_types}"),
            (len(mw_analogs) >= 15, f"At least 15 optimal MW analogs: {len(mw_analogs)}")
        ]
        
        all_passed = True
        for passed, message in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {message}")
            if not passed:
                all_passed = False
        
        print()
        if all_passed:
            print("🎉 ALL TESTS PASSED! Enhanced analog generator is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the implementation.")
        
        return all_passed
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure RDKit is installed: pip install rdkit")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_integration():
    """Test integration with main pipeline."""
    print("\n🔗 Testing Integration with Main Pipeline...")
    print("=" * 50)
    
    try:
        from main import load_config
        from agent_core.data_loader import load_cisplatin_context
        
        # Load context
        cisplatin_context = load_cisplatin_context("data/cisplatin.csv")
        base_smiles = cisplatin_context.get("base_smiles", "N[Pt](N)(Cl)Cl")
        
        print(f"🧬 Base SMILES: {base_smiles}")
        
        # Test enhanced generator with pipeline
        from agent_core.enhanced_analog_generator import generate_enhanced_analogs_30
        analogs = generate_enhanced_analogs_30(base_smiles)
        
        print(f"✅ Integration test passed: {len(analogs)} analogs generated")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Analog Generator Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_enhanced_analog_generator()
    test2_passed = test_integration()
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    print(f"   Enhanced Generator Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Integration Test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your enhanced analog generator is ready for drug discovery!")
        print("   Run: python main.py --mode inference")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    print("=" * 60) 