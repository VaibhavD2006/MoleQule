#!/usr/bin/env python3
"""
Script to add analog names to the cisplatin_analogs.csv file.
Extracts analog names from xyz_path and adds them as a new column.
"""

import pandas as pd
import re
import os
from pathlib import Path

def extract_analog_name(xyz_path):
    """
    Extract analog name from xyz_path.
    
    Args:
        xyz_path (str): Path to XYZ file
        
    Returns:
        str: Extracted analog name
    """
    if pd.isna(xyz_path) or not xyz_path:
        return "Unknown"
    
    # Extract filename from path
    filename = os.path.basename(xyz_path)
    
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Extract analog name from enhanced_analog_*_*.xyz pattern
    if name_without_ext.startswith('enhanced_analog_'):
        # Remove 'enhanced_analog_' prefix and UUID suffix
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            # Join the middle parts to get the analog name
            analog_parts = parts[2:-1]  # Skip 'enhanced', 'analog', and UUID
            analog_name = '_'.join(analog_parts)
            return analog_name
    elif name_without_ext.startswith('analog_'):
        # Remove 'analog_' prefix and UUID suffix
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            analog_name = parts[1]  # Get the part after 'analog_'
            return analog_name
    
    return "Unknown"

def add_analog_names_to_csv(csv_path):
    """
    Add analog names column to the CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    print(f"Processing {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract analog names from xyz_path
    print("Extracting analog names from xyz_path...")
    analog_names = []
    
    for idx, row in df.iterrows():
        # Check if this is a comment row or empty row
        if pd.isna(row.get('analog_id', '')) or str(row.get('analog_id', '')).startswith('#'):
            analog_names.append("")
        else:
            xyz_path = row.get('vqe_descriptors', '')
            if isinstance(xyz_path, str) and 'xyz_path' in xyz_path:
                # Extract xyz_path from the dictionary string
                import ast
                try:
                    descriptors_dict = ast.literal_eval(xyz_path)
                    xyz_path_value = descriptors_dict.get('xyz_path', '')
                    analog_name = extract_analog_name(xyz_path_value)
                    analog_names.append(analog_name)
                except:
                    analog_names.append("Unknown")
            else:
                analog_names.append("Unknown")
    
    # Add the analog_name column after analog_id
    df.insert(1, 'analog_name', analog_names)
    
    # Save the updated CSV
    output_path = csv_path.replace('.csv', '_with_names.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Updated CSV saved to: {output_path}")
    
    # Show some examples
    print("\nExample analog names extracted:")
    for i, name in enumerate(analog_names[:10]):
        if name and name != "Unknown":
            print(f"  {i+1}. {name}")
    
    return output_path

def create_summary_table(csv_path):
    """
    Create a summary table with the top analogs and their names.
    
    Args:
        csv_path (str): Path to the CSV file with names
    """
    df = pd.read_csv(csv_path)
    
    # Filter out comment rows and empty rows
    df_clean = df[df['analog_id'].notna() & ~df['analog_id'].astype(str).str.startswith('#')]
    
    # Sort by final_score (descending)
    df_sorted = df_clean.sort_values('final_score', ascending=False)
    
    # Create summary table
    summary_data = []
    for idx, row in df_sorted.head(20).iterrows():
        analog_name = row.get('analog_name', 'Unknown')
        binding_affinity = row.get('binding_affinity', 0)
        final_score = row.get('final_score', 0)
        
        # Convert score to percentage
        score_percentage = final_score * 100
        
        # Determine grade
        if score_percentage >= 90:
            grade = "A+"
        elif score_percentage >= 80:
            grade = "A"
        elif score_percentage >= 70:
            grade = "B+"
        elif score_percentage >= 60:
            grade = "B"
        else:
            grade = "C"
        
        summary_data.append({
            'Rank': len(summary_data) + 1,
            'Analog': analog_name,
            'Binding (kcal/mol)': f"{binding_affinity:.2f}",
            'Score (%)': f"{score_percentage:.1f}%",
            'Grade': grade
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_path = csv_path.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nSummary table saved to: {summary_path}")
    print("\nTop 10 Analogs:")
    print(summary_df.head(10).to_string(index=False))
    
    return summary_path

if __name__ == "__main__":
    csv_path = "results/cisplatin_analogs.csv"
    
    if os.path.exists(csv_path):
        # Add analog names to the CSV
        updated_csv = add_analog_names_to_csv(csv_path)
        
        # Create summary table
        summary_csv = create_summary_table(updated_csv)
        
        print(f"\n✅ Successfully processed {csv_path}")
        print(f"📊 Added analog names and created summary table")
    else:
        print(f"❌ File not found: {csv_path}") 