import requests
import json

try:
    response = requests.get('http://localhost:8001/health')
    data = response.json()
    
    print("=== ML Service Status ===")
    print(f"Status: {data['status']}")
    print(f"Quantum Dock Available: {data['quantum_dock_available']}")
    print(f"Model Loaded: {data['model_loaded']}")
    print(f"Mode: {data['mode']}")
    
    if data['mode'] == 'quantum':
        print("\n✅ RUNNING IN QUANTUM MODE - Analogs are DYNAMIC!")
        print("   - Real VQE calculations")
        print("   - Real QNN predictions") 
        print("   - Dynamic analog generation from YOUR input")
    else:
        print("\n⚠️  RUNNING IN FALLBACK MODE - Analogs may be static")
        print("   - Mock VQE calculations")
        print("   - Fallback predictions")
        print("   - Limited analog generation")
        
except Exception as e:
    print(f"Error checking status: {e}") 