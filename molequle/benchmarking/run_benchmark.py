#!/usr/bin/env python3
"""
MoleQule Benchmarking - Main Runner
Based on benchmark.md specifications

This script orchestrates the complete benchmarking pipeline.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import yaml
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

class MoleQuleBenchmarker:
    """
    Main benchmarking orchestrator for MoleQule platform
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the benchmarker"""
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(__file__).parent / "results"
        self.data_dir = Path(__file__).parent / "data"
        self.scripts_dir = Path(__file__).parent / "scripts"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Benchmark phases
        self.phases = {
            "phase_1": {
                "name": "Dataset Curation",
                "script": "01_dataset_curation.py",
                "description": "Curate cisplatin analog dataset with experimental data",
                "required": True
            },
            "phase_2": {
                "name": "Docking Method Validation",
                "script": "02_docking_validation.py",
                "description": "Validate all MoleQule docking methods",
                "required": True
            },
            "phase_3": {
                "name": "Comparative Benchmarking",
                "script": "03_comparative_benchmarking.py",
                "description": "Compare against industry standards",
                "required": True
            }
        }
        
        # Benchmark results
        self.benchmark_results = {
            "benchmark_id": f"molequle_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "phases": {},
            "overall_results": {}
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        config_file = Path(__file__).parent / self.config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            return {}
    
    def run_phase(self, phase_key: str) -> Dict[str, Any]:
        """Run a specific benchmark phase"""
        phase_info = self.phases[phase_key]
        script_path = self.scripts_dir / phase_info["script"]
        
        if not script_path.exists():
            raise FileNotFoundError(f"Phase script not found: {script_path}")
        
        print(f"\n{'='*60}")
        print(f"🧬 PHASE: {phase_info['name']}")
        print(f"📝 {phase_info['description']}")
        print(f"📁 Script: {script_path}")
        print(f"{'='*60}")
        
        # Record phase start
        phase_start = time.time()
        
        try:
            # Import and run phase module
            import importlib.util
            spec = importlib.util.spec_from_file_location(phase_key, script_path)
            phase_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(phase_module)
            
            # Run the phase
            if hasattr(phase_module, 'main'):
                phase_module.main()
            
            # Record phase completion
            phase_end = time.time()
            phase_duration = phase_end - phase_start
            
            phase_result = {
                "status": "completed",
                "start_time": phase_start,
                "end_time": phase_end,
                "duration": phase_duration,
                "error": None
            }
            
            print(f"✅ Phase completed successfully in {phase_duration:.2f} seconds")
            
        except Exception as e:
            phase_end = time.time()
            phase_duration = phase_end - phase_start
            
            phase_result = {
                "status": "failed",
                "start_time": phase_start,
                "end_time": phase_end,
                "duration": phase_duration,
                "error": str(e)
            }
            
            print(f"❌ Phase failed after {phase_duration:.2f} seconds")
            print(f"   Error: {e}")
            
            if phase_info["required"]:
                raise e
        
        return phase_result
    
    def run_full_benchmark(self, phases: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete benchmarking pipeline"""
        print("🧬 MoleQule Benchmarking Pipeline")
        print("="*60)
        print(f"🚀 Starting benchmark: {self.benchmark_results['benchmark_id']}")
        print(f"⏰ Start time: {self.benchmark_results['start_time']}")
        print(f"📁 Results directory: {self.results_dir}")
        
        # Determine which phases to run
        if phases is None:
            phases = list(self.phases.keys())
        
        # Validate phases
        for phase in phases:
            if phase not in self.phases:
                raise ValueError(f"Unknown phase: {phase}")
        
        print(f"\n📋 Phases to run: {', '.join(phases)}")
        
        # Run phases
        for phase_key in phases:
            phase_result = self.run_phase(phase_key)
            self.benchmark_results["phases"][phase_key] = phase_result
            
            # Check if phase failed
            if phase_result["status"] == "failed":
                print(f"\n❌ Benchmark failed at phase: {phase_key}")
                break
        
        # Record overall results
        self.benchmark_results["end_time"] = datetime.now().isoformat()
        self.benchmark_results["total_duration"] = time.time() - time.mktime(
            datetime.fromisoformat(self.benchmark_results["start_time"]).timetuple()
        )
        
        # Generate overall assessment
        self.benchmark_results["overall_results"] = self._generate_overall_assessment()
        
        # Save results
        self._save_benchmark_results()
        
        # Generate final report
        self._generate_final_report()
        
        return self.benchmark_results
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall benchmark assessment"""
        completed_phases = [p for p, r in self.benchmark_results["phases"].items() 
                           if r["status"] == "completed"]
        failed_phases = [p for p, r in self.benchmark_results["phases"].items() 
                        if r["status"] == "failed"]
        
        # Load results from completed phases
        assessment = {
            "completion_status": "completed" if len(failed_phases) == 0 else "failed",
            "phases_completed": len(completed_phases),
            "phases_failed": len(failed_phases),
            "success_criteria": {}
        }
        
        # Check if we have enough data for assessment
        if "phase_3" in completed_phases:
            try:
                # Load comparative benchmark results
                benchmark_file = self.results_dir / "comparative_benchmark_results.json"
                if benchmark_file.exists():
                    with open(benchmark_file, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    # Extract success criteria
                    criteria = benchmark_data.get("success_criteria_assessment", {})
                    assessment["success_criteria"] = criteria
                    
                    # Overall success assessment
                    technical_success = criteria.get("technical_performance", {}).get("rmse_target_met", False)
                    experimental_success = criteria.get("experimental_validation", {}).get("r2_target_met", False)
                    commercial_success = criteria.get("commercial_readiness", {}).get("competitive_advantage", False)
                    
                    assessment["overall_success"] = all([
                        technical_success,
                        experimental_success,
                        commercial_success
                    ])
                    
            except Exception as e:
                assessment["success_criteria"] = {"error": f"Could not load results: {e}"}
                assessment["overall_success"] = False
        
        return assessment
    
    def _save_benchmark_results(self):
        """Save benchmark results to file"""
        results_file = self.results_dir / f"{self.benchmark_results['benchmark_id']}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        print(f"\n💾 Benchmark results saved: {results_file}")
    
    def _generate_final_report(self):
        """Generate final benchmark report"""
        print("\n" + "="*60)
        print("MOLECULE BENCHMARKING - FINAL REPORT")
        print("="*60)
        
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   Benchmark ID: {self.benchmark_results['benchmark_id']}")
        print(f"   Start Time: {self.benchmark_results['start_time']}")
        print(f"   End Time: {self.benchmark_results['end_time']}")
        print(f"   Total Duration: {self.benchmark_results['total_duration']:.2f} seconds")
        
        print(f"\n📋 PHASE RESULTS:")
        for phase_key, phase_result in self.benchmark_results["phases"].items():
            phase_info = self.phases[phase_key]
            status_icon = "✅" if phase_result["status"] == "completed" else "❌"
            print(f"   {status_icon} {phase_info['name']}: {phase_result['status']} ({phase_result['duration']:.2f}s)")
            if phase_result["error"]:
                print(f"      Error: {phase_result['error']}")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        overall = self.benchmark_results["overall_results"]
        completion_icon = "✅" if overall["completion_status"] == "completed" else "❌"
        print(f"   Completion Status: {completion_icon} {overall['completion_status']}")
        print(f"   Phases Completed: {overall['phases_completed']}")
        print(f"   Phases Failed: {overall['phases_failed']}")
        
        if "overall_success" in overall:
            success_icon = "✅" if overall["overall_success"] else "❌"
            print(f"   Overall Success: {success_icon} {overall['overall_success']}")
        
        if "success_criteria" in overall and "error" not in overall["success_criteria"]:
            criteria = overall["success_criteria"]
            print(f"\n📈 SUCCESS CRITERIA:")
            
            # Technical Performance
            tech = criteria.get("technical_performance", {})
            tech_icon = "✅" if tech.get("rmse_target_met", False) else "❌"
            print(f"   {tech_icon} RMSE Target (<2.0): {tech.get('rmse_target_met', False)}")
            
            sig_icon = "✅" if tech.get("statistical_significance", False) else "❌"
            print(f"   {sig_icon} Statistical Significance: {tech.get('statistical_significance', False)}")
            
            # Experimental Validation
            exp = criteria.get("experimental_validation", {})
            r2_icon = "✅" if exp.get("r2_target_met", False) else "❌"
            print(f"   {r2_icon} R² Target (>0.75): {exp.get('r2_target_met', False)}")
            
            pose_icon = "✅" if exp.get("pose_accuracy_target_met", False) else "❌"
            print(f"   {pose_icon} Pose Accuracy Target (>0.70): {exp.get('pose_accuracy_target_met', False)}")
            
            # Commercial Readiness
            comm = criteria.get("commercial_readiness", {})
            comp_icon = "✅" if comm.get("competitive_advantage", False) else "❌"
            print(f"   {comp_icon} Competitive Advantage: {comm.get('competitive_advantage', False)}")
            
            if "improvement_percentage" in comm:
                print(f"   📈 Improvement vs Vina: {comm['improvement_percentage']:.1f}%")
        
        print(f"\n📁 GENERATED FILES:")
        for file_path in self.results_dir.glob("*"):
            if file_path.is_file():
                print(f"   {file_path}")
        
        print(f"\n🚀 NEXT STEPS:")
        if overall["completion_status"] == "completed":
            print("   ✅ Benchmark completed successfully!")
            print("   📊 Review results in the results directory")
            print("   📈 Consider Phase 4: Experimental validation")
            print("   📋 Prepare for investor presentation")
        else:
            print("   ❌ Benchmark failed - review errors above")
            print("   🔧 Fix issues and re-run failed phases")
            print("   📞 Contact development team for support")
        
        print("\n" + "="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MoleQule Benchmarking Pipeline")
    parser.add_argument("--phases", nargs="+", 
                       choices=["phase_1", "phase_2", "phase_3"],
                       help="Specific phases to run (default: all)")
    parser.add_argument("--config", default="config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = MoleQuleBenchmarker(config_path=args.config)
    
    # Run benchmark
    try:
        results = benchmarker.run_full_benchmark(phases=args.phases)
        
        if results["overall_results"]["completion_status"] == "completed":
            print(f"\n🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
            return 0
        else:
            print(f"\n❌ BENCHMARK FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 BENCHMARK CRASHED: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 