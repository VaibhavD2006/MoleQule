#!/usr/bin/env python3
"""
Debug script to check file paths and ML service communication
"""

import os
import requests
import json
from pathlib import Path

def debug_paths():
    """Debug file paths and service communication"""
    
    print("=== Path Debugging ===")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check for sample files
    sample_files = ["sample_cisplatin.mol", "sample_cisplatin.smi", "sample_cisplatin.xyz"]
    
    for sample_file in sample_files:
        full_path = os.path.abspath(sample_file)
        exists = os.path.exists(sample_file)
        print(f"File: {sample_file}")
        print(f"  Exists: {exists}")
        print(f"  Full path: {full_path}")
        
        if exists:
            size = os.path.getsize(sample_file)
            print(f"  Size: {size} bytes")
    
    # Check uploads directory
    uploads_dir = "uploads"
    print(f"\nUploads directory: {uploads_dir}")
    print(f"  Exists: {os.path.exists(uploads_dir)}")
    
    if os.path.exists(uploads_dir):
        try:
            for root, dirs, files in os.walk(uploads_dir):
                print(f"  Found: {root}")
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"    File: {file_path}")
        except Exception as e:
            print(f"  Error walking uploads: {e}")
    
    # Test ML service
    print("\n=== ML Service Testing ===")
    
    ml_url = "http://localhost:8001"
    
    # Test health
    try:
        response = requests.get(f"{ml_url}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test with sample file
    sample_file = "sample_cisplatin.mol"
    if os.path.exists(sample_file):
        try:
            request_data = {
                "job_id": "debug_job_123",
                "input_file_path": os.path.abspath(sample_file)
            }
            
            print(f"\nTesting with file: {request_data['input_file_path']}")
            
            response = requests.post(
                f"{ml_url}/process-molecule",
                json=request_data,
                timeout=30
            )
            
            print(f"Process response: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Status: {result.get('status')}")
                print(f"Total analogs: {result.get('total_analogs', 0)}")
            else:
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"Process test failed: {e}")
    else:
        print(f"Sample file not found: {sample_file}")

if __name__ == "__main__":
    debug_paths() 