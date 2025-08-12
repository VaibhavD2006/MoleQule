#!/usr/bin/env python3
"""
Test script for the ML service
"""

import requests
import json
import os

def test_ml_service():
    """Test the ML service endpoints"""
    
    ml_service_url = "http://localhost:8001"
    
    print("Testing ML Service...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{ml_service_url}/health")
        print(f"✓ Health check: {response.status_code}")
        print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test 2: Test endpoint
    try:
        response = requests.post(f"{ml_service_url}/test")
        print(f"✓ Test endpoint: {response.status_code}")
        print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ Test endpoint failed: {e}")
        return False
    
    # Test 3: Process molecule (with sample file)
    sample_file = "sample_cisplatin.mol"
    if os.path.exists(sample_file):
        try:
            request_data = {
                "job_id": "test_job_123",
                "input_file_path": os.path.abspath(sample_file)
            }
            
            response = requests.post(
                f"{ml_service_url}/process-molecule",
                json=request_data,
                timeout=30
            )
            
            print(f"✓ Process molecule: {response.status_code}")
            result = response.json()
            print(f"  Status: {result.get('status')}")
            print(f"  Total analogs: {result.get('total_analogs', 0)}")
            
            if result.get('status') == 'completed':
                analogs = result.get('analogs', [])
                print(f"  First analog: {analogs[0] if analogs else 'None'}")
            
        except Exception as e:
            print(f"✗ Process molecule failed: {e}")
            return False
    else:
        print(f"⚠ Sample file {sample_file} not found, skipping process test")
    
    print("\n✓ All tests completed!")
    return True

if __name__ == "__main__":
    test_ml_service() 