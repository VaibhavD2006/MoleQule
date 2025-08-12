#!/usr/bin/env python3
"""
QuantumDock: A Quantum-Enhanced Drug Discovery Agent
Main entry point for pipeline orchestration and execution.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
import json
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agent_core.data_loader import load_cisplatin_context, load_pancreatic_target
from agent_core.analog_generator import generate_analogs
from vqe_engine.vqe_runner import run_vqe_descriptors
from qnn_model.qnn_predictor import QNNPredictor
from agent_core.scoring_engine import calculate_final_score, rank_analogs


def extract_analog_name(xyz_path: str) -> str:
    """
    Extract analog name from xyz_path.
    
    Args:
        xyz_path (str): Path to XYZ file
        
    Returns:
        str: Extracted analog name
    """
    if not xyz_path:
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


def create_summary_table_with_names(results: List[Dict[str, Any]], output_path: str) -> str:
    """
    Create a summary table with analog names and save to CSV.
    
    Args:
        results (List[Dict[str, Any]]): Results with analog names
        output_path (str): Base output path
        
    Returns:
        str: Path to the summary CSV file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Sort by final_score (descending)
        sorted_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Create summary table
        summary_data = []
        for i, result in enumerate(sorted_results[:20]):  # Top 20 results
            analog_name = result.get('analog_name', 'Unknown')
            binding_affinity = result.get('binding_affinity', 0)
            final_score = result.get('final_score', 0)
            
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
                'Rank': i + 1,
                'Analog': analog_name,
                'Binding (kcal/mol)': f"{binding_affinity:.2f}",
                'Score (%)': f"{score_percentage:.1f}%",
                'Grade': grade
            })
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Summary table saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        logger.error(f"Error creating summary table: {e}")
        return ""


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging system for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quantum_dock.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_training_results_to_csv(training_data: List[List[float]], labels: List[float], 
                                 training_metrics: Dict[str, Any], qnn_config: Dict[str, Any], 
                                 metadata: Dict[str, Any], output_path: str = "results/training_results.csv") -> None:
    """
    Save training results and metadata to CSV file.
    
    Args:
        training_data: Training feature data
        labels: Training labels (binding affinities)
        training_metrics: Final training metrics from QNN
        qnn_config: QNN configuration parameters
        metadata: Training data metadata
        output_path: Path to CSV file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare training results data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(output_path)
        
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Add headers if file is new or empty
            if not file_exists or os.path.getsize(output_path) == 0:
                headers = [
                    "timestamp", "n_samples", "n_features", "n_layers", "epochs", 
                    "learning_rate", "final_mse", "final_mae", "r2_score", 
                    "binding_affinity_min", "binding_affinity_max", "data_source",
                    "feature_names", "training_description"
                ]
                writer.writerow(headers)
            
            # Write training results row
            row = [
                timestamp,
                len(training_data),
                len(training_data[0]) if training_data else 0,
                qnn_config.get("n_layers", "unknown"),
                qnn_config.get("epochs", "unknown"),
                qnn_config.get("learning_rate", "unknown"),
                training_metrics.get("mse", "N/A"),
                training_metrics.get("mae", "N/A"),
                training_metrics.get("r2_score", "N/A"),
                min(labels) if labels else "N/A",
                max(labels) if labels else "N/A",
                metadata.get("source", "unknown"),
                "; ".join(metadata.get("feature_names", [])),
                metadata.get("description", "QNN training session")
            ]
            writer.writerow(row)
            
        logger.info(f"Training results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving training results to CSV: {e}")


def append_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Append new results to existing CSV file with a 5-line spacer, or create new file if it doesn't exist.
    
    Args:
        results (List[Dict[str, Any]]): New results to append
        output_path (str): Path to the CSV file
    """
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.warning("No results to save")
        return
    
    # Create DataFrame from new results
    new_df = pd.DataFrame(results)
    
    # Check if file exists
    if os.path.exists(output_path):
        logger.info(f"Appending {len(results)} new results to existing {output_path}")
        
        # Read existing file
        try:
            existing_df = pd.read_csv(output_path)
            
            # Ensure existing DataFrame has the same columns as new data
            # Add missing columns to existing DataFrame with empty values
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = ""
                    logger.info(f"Added missing column '{col}' to existing CSV")
            
            # Ensure new DataFrame has the same columns as existing data
            # Add missing columns to new DataFrame with empty values
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = ""
                    logger.info(f"Added missing column '{col}' to new data")
            
            # Reorder columns to match existing file
            new_df = new_df[existing_df.columns]
            
            # Create spacer rows (5 empty lines)
            spacer_data = {}
            for col in new_df.columns:
                spacer_data[col] = [""] * 5
            spacer_df = pd.DataFrame(spacer_data)
            
            # Add timestamp comment row
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            comment_data = {}
            for i, col in enumerate(new_df.columns):
                if i == 0:
                    comment_data[col] = f"# New run - {timestamp}"
                else:
                    comment_data[col] = ""
            comment_df = pd.DataFrame([comment_data])
            
            # Combine: existing + spacer + comment + new results
            combined_df = pd.concat([existing_df, spacer_df, comment_df, new_df], ignore_index=True)
            
            # Write back to file
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Successfully appended {len(results)} results with spacer")
            
        except Exception as e:
            logger.error(f"Error reading existing CSV, creating new file: {e}")
            new_df.to_csv(output_path, index=False)
            
    else:
        logger.info(f"Creating new CSV file with {len(results)} results: {output_path}")
        # Add initial timestamp comment
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment_data = {}
        for i, col in enumerate(new_df.columns):
            if i == 0:
                comment_data[col] = f"# Initial run - {timestamp}"
            else:
                comment_data[col] = ""
        comment_df = pd.DataFrame([comment_data])
        
        # Combine comment + results
        final_df = pd.concat([comment_df, new_df], ignore_index=True)
        final_df.to_csv(output_path, index=False)


def run_inference_mode(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run inference mode on new analogs."""
    logger = logging.getLogger(__name__)
    logger.info("Starting inference mode...")
    
    # Load biological context
    cisplatin_context = load_cisplatin_context("data/cisplatin.csv")
    pancreatic_target = load_pancreatic_target("data/protein_binding.csv")
    
    # Generate analogs
    logger.info("Generating molecular analogs...")
    
    # Use enhanced analog generator for 30 pharmaceutical-grade analogs
    try:
        from agent_core.enhanced_analog_generator import generate_enhanced_analogs_30
        analogs = generate_enhanced_analogs_30(cisplatin_context["base_smiles"])
        logger.info(f"✅ Generated {len(analogs)} enhanced pharmaceutical-grade analogs")
    except ImportError:
        # Fallback to basic generator if enhanced version not available
        logger.warning("Enhanced analog generator not available, using basic version")
        analogs = generate_analogs(
            cisplatin_context["base_smiles"],
            cisplatin_context["ligand_substitutions"]
        )
    
    # Initialize QNN predictor
    qnn_config = load_config("configs/qnn_config.yaml")
    qnn_predictor = QNNPredictor(
        n_features=qnn_config["n_features"],
        n_layers=qnn_config["n_layers"],
        n_qubits=qnn_config.get("n_qubits", qnn_config["n_features"]),
        entanglement_type=qnn_config.get("entanglement_type", "linear")
    )
    
    # Load trained model if available
    complete_model_path = "qnn_model/trained_qnn_model.pkl"
    trained_weights_path = "qnn_model/trained_qnn_weights.pkl"
    
    # Try loading complete model first (preferred)
    if os.path.exists(complete_model_path):
        try:
            qnn_predictor.load_model(complete_model_path)
            logger.info(f"Loaded complete trained QNN model from {complete_model_path}")
        except Exception as e:
            logger.warning(f"Could not load complete model: {e}")
            # Fall back to weights-only loading
            if os.path.exists(trained_weights_path):
                try:
                    import pickle
                    with open(trained_weights_path, 'rb') as f:
                        trained_weights = pickle.load(f)
                    qnn_predictor.weights = trained_weights
                    logger.info(f"Loaded trained QNN weights from {trained_weights_path}")
                except Exception as e:
                    logger.warning(f"Could not load trained weights: {e}")
                    logger.info("Using default QNN weights (consider running training mode first)")
            else:
                logger.info("No trained QNN weights found. Using descriptor-based estimation fallback.")
    # Fall back to weights-only if complete model not available
    elif os.path.exists(trained_weights_path):
        try:
            import pickle
            with open(trained_weights_path, 'rb') as f:
                trained_weights = pickle.load(f)
            qnn_predictor.weights = trained_weights
            logger.info(f"Loaded trained QNN weights from {trained_weights_path}")
        except Exception as e:
            logger.warning(f"Could not load trained weights: {e}")
            logger.info("Using default QNN weights (consider running training mode first)")
    else:
        logger.info("No trained QNN model or weights found. Using descriptor-based estimation fallback.")
    
    results = []
    
    # Process each analog
    for i, analog in enumerate(analogs[:config.get("max_analogs", 50)]):
        logger.info(f"Processing analog {i+1}/{len(analogs)}")
        
        try:
            # Run VQE simulation
            vqe_descriptors = run_vqe_descriptors(
                analog["xyz_path"],
                cisplatin_context,
                pancreatic_target
            )
            
            # Predict binding affinity with QNN
            features = [
                vqe_descriptors["energy"],
                vqe_descriptors["homo_lumo_gap"],
                vqe_descriptors["dipole_moment"]
            ]
            
            binding_affinity = qnn_predictor.predict(features)
            
            # Calculate enhanced final score with drug-like properties
            final_score = calculate_final_score(
                binding_affinity,
                vqe_descriptors.get("resistance_score", 0.4),  # Reduced default penalty
                vqe_descriptors.get("toxicity_score", 0.3),    # Reduced default penalty
                vqe_descriptors=vqe_descriptors,               # Pass descriptors for drug-like scoring
                selectivity_bonus=0.1                          # Add selectivity bonus
            )
            
            # Extract analog name from xyz_path
            analog_name = extract_analog_name(analog["xyz_path"])
            
            results.append({
                "analog_id": analog["id"],
                "analog_name": analog_name,
                "smiles": analog["smiles"],
                "binding_affinity": binding_affinity,
                "final_score": final_score,
                "vqe_descriptors": vqe_descriptors
            })
            
        except Exception as e:
            logger.error(f"Error processing analog {i+1}: {str(e)}")
            continue
    
    # Rank analogs
    ranked_results = rank_analogs(results)
    
    logger.info(f"Completed inference on {len(ranked_results)} analogs")
    return ranked_results


def run_training_mode(config: Dict[str, Any]) -> None:
    """Train QNN models on experimental cisplatin analog data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training mode...")
    
    # Load training data from JSON file
    training_data_path = "data/cisplatin_training_data.json"
    
    try:
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data_json = json.load(f)
        
        training_data = training_data_json["features"]
        labels = training_data_json["labels"]
        metadata = training_data_json.get("metadata", {})
        
        logger.info(f"Loaded {len(training_data)} training samples from {training_data_path}")
        logger.info(f"Training data: {metadata.get('description', 'Cisplatin analog binding affinities')}")
        logger.info(f"Features: {metadata.get('feature_names', ['energy', 'homo_lumo_gap', 'dipole_moment'])}")
        logger.info(f"Binding affinity range: {min(labels):.1f} to {max(labels):.1f} kcal/mol")
        
    except FileNotFoundError:
        logger.error(f"Training data file not found: {training_data_path}")
        logger.info("Create the training data file with experimental binding affinity data")
        logger.info("Expected format: {'features': [[energy, gap, dipole], ...], 'labels': [binding_affinity, ...]}")
        return
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return
    
    # Initialize and train QNN
    qnn_config = load_config("configs/qnn_config.yaml")
    qnn_predictor = QNNPredictor(
        n_features=qnn_config["n_features"],
        n_layers=qnn_config["n_layers"],
        n_qubits=qnn_config.get("n_qubits", qnn_config["n_features"]),
        entanglement_type=qnn_config.get("entanglement_type", "linear")
    )
    
    if training_data and labels:
        logger.info(f"Training QNN with {len(training_data)} samples...")
        logger.info(f"Training parameters: epochs={qnn_config.get('epochs', 100)}, lr={qnn_config.get('learning_rate', 0.01)}")
        
        try:
            training_metrics = qnn_predictor.train(
                training_data, 
                labels, 
                epochs=qnn_config.get("epochs", 100),
                learning_rate=qnn_config.get("learning_rate", 0.01)
            )
            
            logger.info("QNN training completed successfully!")
            if training_metrics:
                logger.info(f"Final training metrics: {training_metrics}")
            
            # Save trained model with complete configuration
            model_save_path = "qnn_model/trained_qnn_model.pkl"
            try:
                qnn_predictor.save_model(model_save_path)
                logger.info(f"Complete trained model saved to {model_save_path}")
            except Exception as e:
                logger.warning(f"Could not save complete model: {e}")
                
            # Also save just weights for backward compatibility
            weights_save_path = "qnn_model/trained_qnn_weights.pkl"
            try:
                import pickle
                os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
                with open(weights_save_path, 'wb') as f:
                    pickle.dump(qnn_predictor.weights, f)
                logger.info(f"Trained model weights saved to {weights_save_path}")
            except Exception as e:
                logger.warning(f"Could not save model weights: {e}")
            
            # Save training results to CSV
            if training_metrics:
                try:
                    save_training_results_to_csv(
                        training_data=training_data,
                        labels=labels,
                        training_metrics=training_metrics,
                        qnn_config=qnn_config,
                        metadata=metadata
                    )
                except Exception as e:
                    logger.warning(f"Could not save training results to CSV: {e}")
                
        except Exception as e:
            logger.error(f"Error during QNN training: {e}")
            return
            
    else:
        logger.warning("No training data available - skipping QNN training")


def main():
    """Main entry point for the QuantumDock application."""
    parser = argparse.ArgumentParser(
        description="QuantumDock: Quantum-Enhanced Drug Discovery Agent"
    )
    parser.add_argument(
        "--mode", 
        choices=["inference", "training"],
        default="inference",
        help="Operation mode (default: inference)"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--output",
        default="results/cisplatin_analogs.csv",
        help="Output file path for results (will append to existing file with spacer)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        if args.mode == "inference":
            results = run_inference_mode(config)
            
            # Append results to existing CSV with spacer
            append_results_to_csv(results, args.output)
            logger.info(f"Results appended to {args.output}")
            
            # Create summary table with analog names
            summary_path = create_summary_table_with_names(results, args.output)
            if summary_path:
                logger.info(f"Summary table with analog names created: {summary_path}")
            
        elif args.mode == "training":
            logger.info("Starting training mode...")
            run_training_mode(config)
            logger.info("Training mode completed successfully")
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)
    
    logger.info("QuantumDock execution completed successfully")


if __name__ == "__main__":
    main() 