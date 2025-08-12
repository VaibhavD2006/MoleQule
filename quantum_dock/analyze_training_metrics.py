#!/usr/bin/env python3
"""
Training Metrics Analysis for QuantumDock
Explains what each metric means and calculates accuracy for drug discovery.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt


def analyze_training_metrics():
    """Comprehensive analysis of training metrics and accuracy calculation."""
    
    print("=" * 80)
    print("🧬 QUANTUMDOCK TRAINING METRICS ANALYSIS")
    print("=" * 80)
    
    # Load training results
    try:
        results_df = pd.read_csv('results/training_results.csv')
        latest_run = results_df.iloc[-1]  # Get most recent training run
        
        print(f"\n📊 LATEST TRAINING SESSION: {latest_run['timestamp']}")
        print("-" * 50)
        
    except FileNotFoundError:
        print("❌ No training results found. Run training mode first!")
        return
    
    # Load original training data for context
    with open('data/cisplatin_training_data.json', 'r') as f:
        training_data = json.load(f)
    
    binding_affinities = training_data['labels']
    
    # Extract metrics
    mse = float(latest_run['final_mse'])
    mae = float(latest_run['final_mae'])
    r2 = float(latest_run['r2_score'])
    
    # Calculate additional metrics
    rmse = np.sqrt(mse)
    binding_range = float(latest_run['binding_affinity_max']) - float(latest_run['binding_affinity_min'])
    
    print("\n📈 METRIC DEFINITIONS & INTERPRETATIONS:")
    print("-" * 50)
    
    print(f"""
🎯 MEAN SQUARED ERROR (MSE): {mse:.3f}
   • What it means: Average of squared prediction errors
   • Formula: Σ(predicted - actual)² / n
   • Lower = Better (0 = perfect)
   • Your result: {mse:.3f} kcal²/mol²

📏 MEAN ABSOLUTE ERROR (MAE): {mae:.3f} kcal/mol
   • What it means: Average absolute prediction error
   • Formula: Σ|predicted - actual| / n  
   • Your result: ±{mae:.3f} kcal/mol average error
   • This means predictions are typically off by {mae:.1f} kcal/mol

📐 ROOT MEAN SQUARE ERROR (RMSE): {rmse:.3f} kcal/mol
   • What it means: Standard deviation of prediction errors
   • Formula: √(MSE)
   • Your result: ±{rmse:.3f} kcal/mol typical error

📊 R² SCORE (Coefficient of Determination): {r2:.3f}
   • What it means: Proportion of variance explained by model
   • Range: -∞ to 1.0 (1.0 = perfect, 0 = baseline, negative = worse than baseline)
   • Your result: {r2:.3f} = Model explains {max(0, r2*100):.1f}% of variance
   • ⚠️  Negative R² means model performs worse than simply predicting the mean
""")
    
    print("\n🎯 ACCURACY CALCULATIONS FOR DRUG DISCOVERY:")
    print("-" * 50)
    
    # Calculate accuracy percentages
    mean_binding = np.mean(binding_affinities)
    
    # Accuracy within different tolerance levels
    tolerances = [0.5, 1.0, 1.5, 2.0]
    
    print(f"📊 Binding Affinity Context:")
    print(f"   • Range: {min(binding_affinities):.1f} to {max(binding_affinities):.1f} kcal/mol")
    print(f"   • Mean: {mean_binding:.1f} kcal/mol") 
    print(f"   • Standard deviation: {np.std(binding_affinities):.1f} kcal/mol")
    print(f"   • Total range: {binding_range:.1f} kcal/mol")
    
    print(f"\n🎯 ESTIMATED PREDICTION ACCURACY:")
    print(f"   (Based on MAE = {mae:.1f} kcal/mol)")
    
    for tol in tolerances:
        # Approximate accuracy assuming normal distribution of errors
        accuracy = (1 - mae/tol) * 100 if mae < tol else 0
        accuracy = max(0, min(100, accuracy))
        
        print(f"   • Within ±{tol} kcal/mol: ~{accuracy:.0f}% of predictions")
    
    print(f"\n🧪 PHARMACEUTICAL CONTEXT:")
    print("-" * 50)
    
    # Industry standards context
    print(f"""
📋 DRUG DISCOVERY STANDARDS:
   • Excellent: MAE < 0.5 kcal/mol
   • Good: MAE 0.5-1.0 kcal/mol  
   • Acceptable: MAE 1.0-1.5 kcal/mol
   • Poor: MAE > 1.5 kcal/mol
   
   YOUR MODEL: MAE = {mae:.1f} kcal/mol → {"EXCELLENT" if mae < 0.5 else "GOOD" if mae < 1.0 else "ACCEPTABLE" if mae < 1.5 else "NEEDS IMPROVEMENT"}

🔬 BINDING AFFINITY SIGNIFICANCE:
   • 1 kcal/mol difference = ~5x change in binding strength
   • 2 kcal/mol difference = ~25x change in binding strength  
   • Clinical significance: Usually need <1 kcal/mol accuracy

💊 YOUR MODEL'S CLINICAL RELEVANCE:
   • Typical error: ±{mae:.1f} kcal/mol
   • This represents ~{np.exp(mae/0.6):.0f}x uncertainty in binding strength
   • {"✅ Clinically useful" if mae < 1.5 else "⚠️ Limited clinical utility"}
""")
    
    print(f"\n🚀 IMPROVEMENT RECOMMENDATIONS:")
    print("-" * 50)
    
    if r2 < 0:
        print("❌ CRITICAL: Negative R² indicates model is worse than baseline!")
        print("   Recommendations:")
        print("   • Increase training epochs (current: 100)")
        print("   • Try different learning rates")
        print("   • Add more training data")
        print("   • Check for data quality issues")
        
    elif mae > 1.5:
        print("⚠️  HIGH ERROR: Model needs significant improvement")
        print("   Recommendations:")
        print("   • Increase model complexity (more layers/features)")
        print("   • Gather more diverse training data")
        print("   • Feature engineering (normalize inputs)")
        
    elif mae > 1.0:
        print("📈 MODERATE ERROR: Model shows promise but can improve")
        print("   Recommendations:")
        print("   • Fine-tune hyperparameters")
        print("   • Add validation data")
        print("   • Implement cross-validation")
        
    else:
        print("✅ GOOD PERFORMANCE: Model is performing well!")
        print("   Suggestions:")
        print("   • Test on external validation set")
        print("   • Deploy for drug screening")
        print("   • Document results for publication")
    
    print(f"\n📊 TRAINING HISTORY:")
    print("-" * 50)
    
    if len(results_df) > 1:
        print("Training session comparison:")
        for i, row in results_df.iterrows():
            print(f"   {i+1}. {row['timestamp']}: MAE={row['final_mae']:.3f}, R²={row['r2_score']:.3f}")
        
        # Calculate improvement
        if len(results_df) >= 2:
            prev_mae = float(results_df.iloc[-2]['final_mae'])
            improvement = ((prev_mae - mae) / prev_mae) * 100
            print(f"\n   📈 Recent improvement: {improvement:+.1f}% change in MAE")
    else:
        print("   Only one training session recorded.")
    
    print("\n" + "=" * 80)
    print("💡 TIP: Run more training sessions to track improvement over time!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_training_metrics() 