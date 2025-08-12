#!/usr/bin/env python3
"""
Test script for comprehensive analysis API
"""

import requests
import json

def test_comprehensive_analysis():
    """Test the comprehensive analysis endpoint"""
    
    url = "http://localhost:8000/api/v1/comprehensive-analysis"
    data = {
        "smiles": "N[Pt](N)(F)F"
    }
    
    try:
        print("Testing comprehensive analysis endpoint...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API call successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection error - backend service might not be running")
    except requests.exceptions.Timeout:
        print("✗ Request timeout")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def test_health_endpoint():
    """Test the health endpoint"""
    
    url = "http://localhost:8000/health"
    
    try:
        print("\nTesting health endpoint...")
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Health check successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection error - backend service might not be running")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_health_endpoint()
    test_comprehensive_analysis() 