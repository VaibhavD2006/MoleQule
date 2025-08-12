#!/usr/bin/env python3
"""
Test script to verify that docking visualization is dynamic and model-driven
"""
import requests
import json
import time

def test_dynamic_docking():
    print('🔍 TESTING DYNAMIC DOCKING PIPELINE')
    print('='*60)
    
    # Test 1: Check service capabilities
    print('\n1. Checking service capabilities...')
    try:
        response = requests.get('http://localhost:8002/health')
        if response.status_code == 200:
            health = response.json()
            print(f'   ✅ Service Status: {health.get("status", "unknown")}')
            print(f'   ✅ RDKit Available: {health.get("rdkit_available", False)}')
            print(f'   ✅ Py3Dmol Available: {health.get("py3dmol_available", False)}')
            print(f'   ✅ Advanced Docking: {health.get("advanced_docking_available", False)}')
            
            if not all([health.get("rdkit_available"), health.get("py3dmol_available")]):
                print('   ❌ WARNING: Missing required libraries for dynamic visualization')
        else:
            print(f'   ❌ Service not responding: {response.status_code}')
            return False
    except Exception as e:
        print(f'   ❌ Cannot connect to docking service: {e}')
        return False
    
    # Test 2: Test different molecules for dynamic behavior
    print('\n2. Testing different molecules for dynamic behavior...')
    test_molecules = [
        {'name': 'Cisplatin', 'smiles': 'N[Pt](N)(Cl)Cl'},
        {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'},
        {'name': 'Aspirin', 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'}
    ]
    
    results = []
    for i, mol in enumerate(test_molecules, 1):
        print(f'\n   Test {i}: {mol["name"]} ({mol["smiles"]})')
        
        request_data = {
            'analog_id': f'test_dynamic_{i}',
            'analog_smiles': mol['smiles'],
            'target_protein': 'DNA',
            'method': 'quantum'
        }
        
        try:
            response = requests.post('http://localhost:8002/dock-molecule', 
                                   json=request_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                binding_score = result.get('binding_score', 0)
                method_used = result.get('method_used', 'unknown')
                interactions = result.get('interactions', [])
                viz_html = result.get('visualization_html', '')
                
                print(f'      Binding Score: {binding_score} kcal/mol')
                print(f'      Method Used: {method_used}')
                print(f'      Interactions: {len(interactions)} detected')
                print(f'      Visualization: {len(viz_html)} chars')
                
                # Check for dynamic indicators
                is_dynamic_smiles = mol['smiles'] in viz_html
                is_dynamic_3d = '3Dmol.org' in viz_html and 'viewer.addModel' in viz_html
                has_mol_data = 'viewer.addModel(' in viz_html
                
                if is_dynamic_smiles:
                    print('      ✅ DYNAMIC: Contains input SMILES in visualization')
                else:
                    print('      ⚠️  STATIC?: Missing input SMILES in visualization')
                    
                if is_dynamic_3d:
                    print('      ✅ DYNAMIC: Contains 3D molecular viewer components')
                else:
                    print('      ❌ STATIC: Missing 3D viewer components')
                    
                if has_mol_data:
                    print('      ✅ DYNAMIC: Contains molecular structure data')
                else:
                    print('      ❌ STATIC: Missing molecular structure data')
                
                results.append({
                    'molecule': mol['name'],
                    'smiles': mol['smiles'],
                    'binding_score': binding_score,
                    'method': method_used,
                    'interactions_count': len(interactions),
                    'viz_length': len(viz_html),
                    'is_dynamic': is_dynamic_smiles and is_dynamic_3d and has_mol_data
                })
                
            else:
                print(f'      ❌ Request failed: {response.status_code}')
                print(f'      Error: {response.text[:200]}')
                
        except Exception as e:
            print(f'      ❌ Error: {str(e)}')
    
    # Test 3: Analyze results for dynamic behavior
    print('\n3. DYNAMIC BEHAVIOR ANALYSIS:')
    print('='*60)
    
    if not results:
        print('❌ No successful tests - cannot verify dynamic behavior')
        return False
    
    # Check score variation
    binding_scores = [r['binding_score'] for r in results]
    unique_scores = len(set(binding_scores))
    
    print(f'Binding Scores: {binding_scores}')
    print(f'Unique Scores: {unique_scores}/{len(results)}')
    
    if unique_scores > 1:
        print('✅ CONFIRMED: Different molecules produce different binding scores (DYNAMIC)')
    else:
        print('❌ WARNING: All molecules have identical binding scores (possibly STATIC)')
    
    # Check visualization variation
    viz_lengths = [r['viz_length'] for r in results]
    unique_viz = len(set(viz_lengths))
    
    print(f'Visualization Lengths: {viz_lengths}')
    print(f'Unique Visualizations: {unique_viz}/{len(results)}')
    
    if unique_viz > 1:
        print('✅ CONFIRMED: Different molecules produce different visualizations (DYNAMIC)')
    else:
        print('⚠️  WARNING: All visualizations have same length (check if truly dynamic)')
    
    # Overall assessment
    dynamic_count = sum(1 for r in results if r['is_dynamic'])
    print(f'\nDynamic Tests Passed: {dynamic_count}/{len(results)}')
    
    if dynamic_count == len(results):
        print('🎉 SUCCESS: All tests confirm DYNAMIC behavior!')
        return True
    elif dynamic_count > 0:
        print('⚠️  PARTIAL: Some tests show dynamic behavior, others may be static')
        return False
    else:
        print('❌ FAILURE: All tests suggest STATIC fallback behavior')
        return False

if __name__ == '__main__':
    success = test_dynamic_docking()
    if success:
        print('\n✅ VERIFICATION COMPLETE: Docking visualization is DYNAMIC')
    else:
        print('\n❌ VERIFICATION FAILED: Docking visualization may be using STATIC fallbacks') 