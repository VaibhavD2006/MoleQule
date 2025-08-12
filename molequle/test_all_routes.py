#!/usr/bin/env python3
"""
Test to check all available routes in the API
"""

import requests
import json

def test_all_routes():
    """Test all possible comprehensive analysis routes"""
    
    base_url = "http://localhost:8000"
    
    # Test different possible endpoints
    endpoints = [
        "/api/v1/comprehensive-analysis",
        "/comprehensive-analysis",
        "/api/v1/enhanced/comprehensive-analysis",
        "/enhanced/comprehensive-analysis",
        "/api/v1/health",
        "/health"
    ]
    
    print("Testing all possible routes...")
    
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            if "comprehensive-analysis" in endpoint:
                response = requests.post(url, json={"smiles": "N[Pt](N)(F)F"}, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            print(f"{endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✓ Working!")
            elif response.status_code == 422:
                print(f"  ✓ Route exists (validation error)")
            elif response.status_code == 404:
                print(f"  ✗ Not found")
            else:
                print(f"  ? Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"{endpoint}: Connection error")
        except Exception as e:
            print(f"{endpoint}: Error - {e}")

if __name__ == "__main__":
    test_all_routes() 