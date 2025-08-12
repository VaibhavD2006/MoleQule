#!/usr/bin/env python3
"""Simple Vina Test"""
import requests
import json

def test_vina():
    print('🧬 TESTING DIRECT VINA METHOD')
    print('='*40)

    # Test direct vina method
    response = requests.post('http://localhost:8002/dock-molecule', json={
        'analog_id': 'vina_direct_test',
        'analog_smiles': 'N[Pt](N)(Cl)Cl',
        'target_protein': 'DNA',
        'method': 'vina'
    }, timeout=30)

    if response.status_code == 200:
        data = response.json()
        print(f'✅ Status: Success')
        print(f'✅ Method Used: {data["method_used"]}')
        print(f'✅ Binding Score: {data["binding_score"]:.2f} kcal/mol')
        print(f'✅ Interactions: {len(data["interactions"])}')
        
        # Check if this is real Vina or simulation
        method = data['method_used']
        if 'vina' in method.lower():
            if 'simulation' in method.lower() or 'advanced' in method.lower():
                print('🎭 Using: Advanced Vina Simulation')
            else:
                print('🎯 Using: Real AutoDock Vina')
        else:
            print('⚙️ Using: Fallback method')
            
    else:
        print(f'❌ Failed: {response.status_code}')

if __name__ == "__main__":
    test_vina() 