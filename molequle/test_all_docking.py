#!/usr/bin/env python3
"""
Comprehensive Test for All Docking Methods in MoleQule
Tests: Basic Analysis, QAOA, Classical (Force Field, AutoDock), Binding Sites
"""

import requests
import time
import json
from typing import Dict, Any

def test_docking_service():
    """Test all docking methods and features"""
    
    print("🧬 COMPREHENSIVE DOCKING SERVICE TEST")
    print("=" * 60)
    
    base_url = "http://localhost:8002"
    
    # Test molecules
    test_molecules = [
        {
            "name": "Cisplatin",
            "smiles": "N[Pt](N)(Cl)Cl",
            "analog_id": "cisplatin_original"
        },
        {
            "name": "Carboplatin Analog", 
            "smiles": "N[Pt](N)(Br)Br",
            "analog_id": "carboplatin_analog_br"
        },
        {
            "name": "Enhanced Oxaliplatin",
            "smiles": "N[Pt](NCCN)(O)O",
            "analog_id": "enhanced_oxaliplatin"
        }
    ]
    
    # Test targets
    test_targets = ["DNA", "GSTP1", "p53"]
    
    # Test methods
    test_methods = ["basic", "qaoa", "classical"]
    
    results_summary = {
        "health_check": False,
        "basic_docking": 0,
        "qaoa_docking": 0,
        "classical_docking": 0,
        "binding_sites": 0,
        "visualizations": 0,
        "errors": []
    }
    
    try:
        # 1. Health Check
        print("\n1️⃣ HEALTH CHECK")
        print("-" * 30)
        
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Service Status: {health_data['status']}")
            print(f"✅ Version: {health_data['version']}")
            print(f"✅ Visualization Available: {health_data['visualization_available']}")
            print(f"✅ Advanced Docking Available: {health_data['advanced_docking_available']}")
            
            print("\n🔧 Available Features:")
            for feature, status in health_data['features'].items():
                print(f"   {feature}: {'✅' if status else '❌'}")
            
            print("\n🧪 Docking Methods:")
            for method, status in health_data['docking_methods'].items():
                print(f"   {method}: {status}")
            
            results_summary["health_check"] = True
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return results_summary
        
        # 2. Test Binding Sites
        print("\n2️⃣ BINDING SITE DETECTION")
        print("-" * 35)
        
        for target in test_targets:
            try:
                site_response = requests.get(f"{base_url}/binding-sites/{target}", timeout=10)
                if site_response.status_code == 200:
                    site_data = site_response.json()
                    print(f"✅ {target}: {site_data.get('description', 'Unknown')}")
                    results_summary["binding_sites"] += 1
                else:
                    print(f"❌ {target}: Failed to get binding site")
            except Exception as e:
                print(f"❌ {target}: Error - {e}")
                results_summary["errors"].append(f"Binding site {target}: {e}")
        
        # 3. Test All Docking Methods
        print("\n3️⃣ DOCKING METHODS TEST")
        print("-" * 30)
        
        for method in test_methods:
            print(f"\n🔬 Testing {method.upper()} Docking:")
            method_success = 0
            
            for molecule in test_molecules:
                for target in test_targets:
                    try:
                        print(f"  Testing {molecule['name']} → {target} ({method})")
                        
                        docking_request = {
                            "analog_id": f"{molecule['analog_id']}_{target}_{method}",
                            "analog_smiles": molecule["smiles"],
                            "target_protein": target,
                            "method": method
                        }
                        
                        dock_response = requests.post(
                            f"{base_url}/dock-molecule", 
                            json=docking_request, 
                            timeout=30
                        )
                        
                        if dock_response.status_code == 200:
                            dock_data = dock_response.json()
                            
                            binding_score = dock_data["binding_score"]
                            num_interactions = len(dock_data["interactions"])
                            method_used = dock_data["method_used"]
                            viz_length = len(dock_data["visualization_html"])
                            
                            print(f"    ✅ Score: {binding_score:.2f} kcal/mol | "
                                  f"Interactions: {num_interactions} | "
                                  f"Method: {method_used} | "
                                  f"Viz: {viz_length} chars")
                            
                            method_success += 1
                            results_summary["visualizations"] += 1
                            
                            # Validate interactions
                            if dock_data["interactions"]:
                                interaction_types = [i.get("type", "unknown") for i in dock_data["interactions"]]
                                print(f"    🔗 Interactions: {', '.join(set(interaction_types))}")
                            
                        else:
                            print(f"    ❌ Docking failed: {dock_response.status_code}")
                            results_summary["errors"].append(f"{method} docking {molecule['name']}->{target}: {dock_response.status_code}")
                    
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        results_summary["errors"].append(f"{method} docking {molecule['name']}->{target}: {e}")
                
                # Small delay between tests
                time.sleep(0.5)
            
            # Update method-specific counters
            if method == "basic":
                results_summary["basic_docking"] = method_success
            elif method == "qaoa":
                results_summary["qaoa_docking"] = method_success
            elif method == "classical":
                results_summary["classical_docking"] = method_success
            
            print(f"  📊 {method.upper()} Success Rate: {method_success}/{len(test_molecules) * len(test_targets)}")
        
        # 4. Test Visualization
        print("\n4️⃣ STANDALONE VISUALIZATION")
        print("-" * 35)
        
        try:
            viz_request = {
                "ligand_data": "N[Pt](N)(Cl)Cl",
                "target_data": "DNA binding site"
            }
            
            viz_response = requests.post(
                f"{base_url}/visualize-molecule",
                json=viz_request,
                timeout=15
            )
            
            if viz_response.status_code == 200:
                viz_data = viz_response.json()
                print(f"✅ Visualization created: {len(viz_data['visualization_html'])} chars")
                print(f"✅ Method: {viz_data['method']}")
                results_summary["visualizations"] += 1
            else:
                print(f"❌ Visualization failed: {viz_response.status_code}")
        
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            results_summary["errors"].append(f"Standalone visualization: {e}")
        
        # 5. Performance Test
        print("\n5️⃣ PERFORMANCE TEST")
        print("-" * 25)
        
        start_time = time.time()
        
        performance_request = {
            "analog_id": "performance_test",
            "analog_smiles": "N[Pt](N)(Cl)Cl",
            "target_protein": "DNA",
            "method": "basic"
        }
        
        perf_response = requests.post(
            f"{base_url}/dock-molecule",
            json=performance_request,
            timeout=30
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if perf_response.status_code == 200:
            print(f"✅ Performance test completed in {duration:.2f} seconds")
        else:
            print(f"❌ Performance test failed: {perf_response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to docking service. Is it running on port 8002?")
        results_summary["errors"].append("Service connection failed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        results_summary["errors"].append(f"Unexpected error: {e}")
    
    # Print Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = (
        results_summary["basic_docking"] + 
        results_summary["qaoa_docking"] + 
        results_summary["classical_docking"]
    )
    
    print(f"✅ Health Check: {'PASS' if results_summary['health_check'] else 'FAIL'}")
    print(f"✅ Binding Sites: {results_summary['binding_sites']}/{len(test_targets)}")
    print(f"✅ Basic Docking: {results_summary['basic_docking']}/{len(test_molecules) * len(test_targets)}")
    print(f"✅ QAOA Docking: {results_summary['qaoa_docking']}/{len(test_molecules) * len(test_targets)}")
    print(f"✅ Classical Docking: {results_summary['classical_docking']}/{len(test_molecules) * len(test_targets)}")
    print(f"✅ Visualizations: {results_summary['visualizations']}")
    print(f"✅ Total Successful Tests: {total_tests + results_summary['binding_sites'] + results_summary['visualizations']}")
    
    if results_summary["errors"]:
        print(f"\n❌ Errors ({len(results_summary['errors'])}):")
        for error in results_summary["errors"][:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(results_summary["errors"]) > 5:
            print(f"   • ... and {len(results_summary['errors']) - 5} more errors")
    else:
        print("\n🎉 ALL TESTS PASSED! No errors detected.")
    
    return results_summary

if __name__ == "__main__":
    results = test_docking_service()
    
    # Exit with appropriate code
    if results["health_check"] and not results["errors"]:
        print("\n🎯 Docking service is fully functional!")
        exit(0)
    else:
        print("\n⚠️ Some issues detected in docking service")
        exit(1) 