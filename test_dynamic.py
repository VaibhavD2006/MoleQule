#!/usr/bin/env python3
"""Test script to verify dynamic docking behavior"""

import requests
import time

def test_dynamic_docking():
    print('🧪 TESTING DYNAMIC DOCKING BEHAVIOR')
    print('='*50)
    
    # Wait for service to start
    time.sleep(2)
    
    # Test different molecules
    molecules = [
        {'name': 'Cisplatin', 'smiles': 'N[Pt](N)(Cl)Cl'},
        {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'}
    ]
    
    results = []
    
    for i, mol in enumerate(molecules, 1):
        print(f'\nTest {i}: {mol["name"]}')
        print(f'SMILES: {mol["smiles"]}')
        
        try:
            response = requests.post('http://localhost:8002/dock-molecule', json={
                'analog_id': f'test_{i}',
                'analog_smiles': mol['smiles'],
                'target_protein': 'DNA',
                'method': 'quantum'
            }, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                
                binding_score = result.get('binding_score', 0)
                method_used = result.get('method_used', 'unknown')
                viz_html = result.get('visualization_html', '')
                
                print(f'  Binding Score: {binding_score:.2f} kcal/mol')
                print(f'  Method: {method_used}')
                print(f'  Visualization: {len(viz_html)} chars')
                
                # Check for dynamic indicators
                has_smiles = mol['smiles'] in viz_html
                has_3d = '3Dmol.org' in viz_html
                has_props = 'Molecular Weight:' in viz_html
                
                print(f'  Contains SMILES: {"✅" if has_smiles else "❌"}')
                print(f'  Has 3D viewer: {"✅" if has_3d else "❌"}')
                print(f'  Has properties: {"✅" if has_props else "❌"}')
                
                is_dynamic = has_smiles and (has_3d or has_props)
                print(f'  STATUS: {"✅ DYNAMIC" if is_dynamic else "❌ STATIC"}')
                
                results.append({
                    'name': mol['name'],
                    'score': binding_score,
                    'method': method_used,
                    'is_dynamic': is_dynamic
                })
                
            else:
                print(f'  ❌ Failed: {response.status_code}')
                
        except Exception as e:
            print(f'  ❌ Error: {e}')
    
    # Analysis
    print('\n📊 ANALYSIS:')
    print('='*30)
    
    if len(results) >= 2:
        scores = [r['score'] for r in results]
        dynamic_count = sum(1 for r in results if r['is_dynamic'])
        
        print(f'Scores: {scores}')
        print(f'Different scores: {"✅" if len(set(scores)) > 1 else "❌"}')
        print(f'Dynamic results: {dynamic_count}/{len(results)}')
        
        if dynamic_count == len(results) and len(set(scores)) > 1:
            print('\n🎉 SUCCESS: Fully dynamic docking!')
        elif dynamic_count > 0:
            print('\n⚠️ PARTIAL: Some dynamic behavior')
        else:
            print('\n❌ STATIC: Using fallbacks')
    else:
        print('❌ Not enough results to analyze')

if __name__ == '__main__':
    test_dynamic_docking() 