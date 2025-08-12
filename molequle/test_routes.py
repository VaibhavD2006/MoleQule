#!/usr/bin/env python3
"""
Test script to check available API routes
"""

import requests
import json

def test_available_routes():
    """Test what routes are available"""
    
    base_url = "http://localhost:8000"
    
    # Test different possible comprehensive analysis endpoints
    endpoints = [
        "/api/v1/comprehensive-analysis",
        "/comprehensive-analysis", 
        "/api/v1/enhanced/comprehensive-analysis",
        "/enhanced/comprehensive-analysis"
    ]
    
    print("Testing available routes...")
    
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            response = requests.post(url, json={"smiles": "N[Pt](N)(F)F"}, timeout=5)
            print(f"✓ {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.json()}")
            elif response.status_code == 422:
                print(f"  Validation error (expected): {response.json()}")
        except requests.exceptions.ConnectionError:
            print(f"✗ {endpoint}: Connection error")
        except Exception as e:
            print(f"✗ {endpoint}: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(base_url + "/health", timeout=5)
        print(f"\nHealth endpoint: {response.status_code}")
    except Exception as e:
        print(f"Health endpoint error: {e}")

if __name__ == "__main__":
    test_available_routes() 