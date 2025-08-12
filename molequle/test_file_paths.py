#!/usr/bin/env python3
"""
Test script to verify file paths and upload functionality
"""

import os
import requests
import json
from pathlib import Path

def test_file_paths():
    """Test file path handling"""
    
    print("=== File Path Testing ===")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right place
    if not os.path.exists("backend") or not os.path.exists("ml_service"):
        print("❌ Not in molequle directory. Please run from molequle folder.")
        return
    
    # Test upload directory creation
    upload_dir = os.path.join(current_dir, "uploads")
    print(f"Upload directory: {upload_dir}")
    
    # Create test file
    test_file = os.path.join(upload_dir, "test_job", "test.mol")
    test_dir = os.path.dirname(test_file)
    
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simple test file
    with open(test_file, "w") as f:
        f.write("Test molecule file\n")
    
    print(f"Created test file: {test_file}")
    print(f"File exists: {os.path.exists(test_file)}")
    
    # Test ML service with this file
    print("\n=== Testing ML Service ===")
    
    ml_url = "http://localhost:8001"
    
    # Test health first
    try:
        response = requests.get(f"{ml_url}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✅ ML service is running")
        else:
            print("❌ ML service health check failed")
            return
    except Exception as e:
        print(f"❌ ML service not accessible: {e}")
        return
    
    # Test with our test file
    try:
        request_data = {
            "job_id": "test_job_123",
            "input_file_path": os.path.abspath(test_file)
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
            print(f"✅ Success! Status: {result.get('status')}")
            print(f"Total analogs: {result.get('total_analogs', 0)}")
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Process test failed: {e}")
    
    # Clean up
    try:
        os.remove(test_file)
        os.rmdir(test_dir)
        print(f"\n✅ Cleaned up test files")
    except:
        pass

def test_backend_upload():
    """Test backend upload functionality"""
    
    print("\n=== Backend Upload Testing ===")
    
    backend_url = "http://localhost:8000"
    
    # Test backend health
    try:
        response = requests.get(f"{backend_url}/", timeout=5)
        print(f"Backend health: {response.status_code}")
        if response.status_code != 200:
            print("❌ Backend not accessible")
            return
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return
    
    # Create a test file to upload
    test_content = """Cisplatin
 12 12  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Pt  0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 Cl 0  0  0  0  0  0  0  0  0  0  0  0
   -1.0000    0.0000    0.0000 Cl 0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 N  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.0000    0.0000 N  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END"""
    
    # Test upload
    try:
        files = {'file': ('test_cisplatin.mol', test_content, 'chemical/x-mol')}
        
        response = requests.post(
            f"{backend_url}/api/v1/upload-molecule",
            files=files,
            timeout=30
        )
        
        print(f"Upload response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful! Job ID: {result.get('job_id')}")
            
            # Test getting results
            job_id = result.get('job_id')
            if job_id:
                print(f"\nTesting results for job: {job_id}")
                
                response = requests.get(f"{backend_url}/api/v1/results/{job_id}", timeout=30)
                print(f"Results response: {response.status_code}")
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"✅ Results retrieved! Status: {results.get('status')}")
                    if results.get('analogs'):
                        print(f"Found {len(results['analogs'])} analogs")
                else:
                    print(f"❌ Results error: {response.text}")
        else:
            print(f"❌ Upload failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")

if __name__ == "__main__":
    test_file_paths()
    test_backend_upload() 