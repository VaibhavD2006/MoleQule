#!/usr/bin/env python3
"""
Simple test to check if the comprehensive analysis route is available
"""

import requests
import json

def test_simple_route():
    """Test the comprehensive analysis route with a simple request"""
    
    url = "http://localhost:8000/api/v1/comprehensive-analysis"
    data = {
        "smiles": "N[Pt](N)(F)F"
    }
    
    try:
        print(f"Testing URL: {url}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✓ Route is working!")
        elif response.status_code == 422:
            print("✓ Route exists but validation failed (expected)")
        else:
            print(f"✗ Route returned {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection error - backend not running")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_simple_route() 