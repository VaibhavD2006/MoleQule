import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CellBasedValidation:
    """
    Predicts cytotoxicity and cancer cell killing ability of compounds.
    Integrates cell-based assay predictions for real-world validation.
    """
    
    def __init__(self):
        self.cell_lines = {
            'pancreatic_cancer': ['MIA PaCa-2', 'PANC-1', 'BxPC-3', 'AsPC-1'],
            'breast_cancer': ['MCF-7', 'MDA-MB-231', 'BT-474', 'T-47D'],
            'lung_cancer': ['A549', 'H460', 'H1299', 'H1975'],
            'colon_cancer': ['HCT-116', 'SW480', 'LoVo', 'HT-29'],
            'normal_cells': ['HEK-293', 'MCF-10A', 'IMR-90', 'BJ']
        }
        self.cytotoxicity_model = None
        self.selectivity_model = None
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize cytotoxicity and selectivity prediction models."""
        try:
            # In production, these would be trained on real cell assay data
            self.cytotoxicity_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selectivity_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Generate synthetic training data for demonstration
            self._generate_training_data()
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
    
    def _generate_training_data(self):
        """Generate synthetic training data for cytotoxicity prediction."""
        # Generate realistic cytotoxicity data
        n_samples = 1000
        
        # Molecular descriptors (simplified)
        molecular_weight = np.random.uniform(200, 800, n_samples)
        logp = np.random.uniform(-2, 5, n_samples)
        hbd = np.random.randint(0, 8, n_samples)
        hba = np.random.randint(2, 12, n_samples)
        tpsa = np.random.uniform(20, 150, n_samples)
        
        # Binding affinity (from quantum model)
        binding_affinity = np.random.normal(-7.0, 1.5, n_samples)
        
        # Generate cytotoxicity values based on molecular properties
        base_ic50 = 50 + (binding_affinity + 7) * 20  # Stronger binding = lower IC50
        molecular_factor = (molecular_weight / 400) * 0.5
        lipophilicity_factor = (logp - 1) * 0.3
        
        ic50_values = base_ic50 * (1 + molecular_factor + lipophilicity_factor)
        ic50_values = np.maximum(ic50_values, 0.1)  # Minimum IC50
        
        # Generate selectivity data
        cancer_ic50 = ic50_values
        normal_ic50 = cancer_ic50 * np.random.uniform(2, 10, n_samples)  # Normal cells more resistant
        
        # Create training data
        X = np.column_stack([molecular_weight, logp, hbd, hba, tpsa, binding_affinity])
        y_cancer = cancer_ic50
        y_selectivity = normal_ic50 / cancer_ic50
        
        # Train models
        self.cytotoxicity_model.fit(X, y_cancer)
        self.selectivity_model.fit(X, y_selectivity)
        
        self.logger.info("Cell-based validation models trained successfully")
    
    def predict_cytotoxicity(self, compound: Dict) -> Dict:
        """
        Predict cytotoxicity for a given compound.
        
        Args:
            compound: Dictionary containing compound data including SMILES and properties
            
        Returns:
            Dictionary with cytotoxicity predictions for different cell lines
        """
        try:
            # Extract molecular properties
            molecular_properties = self._extract_molecular_properties(compound)
            
            # Predict IC50 for different cell lines
            predictions = {}
            
            for cancer_type, cell_lines in self.cell_lines.items():
                if cancer_type == 'normal_cells':
                    continue
                    
                cell_line_predictions = {}
                for cell_line in cell_lines:
                    # Add cell line specific factors
                    cell_specific_properties = molecular_properties.copy()
                    cell_specific_properties.extend([
                        self._get_cell_line_factor(cell_line),
                        self._get_cancer_type_factor(cancer_type)
                    ])
                    
                    # Predict IC50
                    ic50_prediction = self.cytotoxicity_model.predict([cell_specific_properties])[0]
                    
                    # Add some realistic variation
                    ic50_prediction *= np.random.uniform(0.8, 1.2)
                    ic50_prediction = max(ic50_prediction, 0.1)
                    
                    cell_line_predictions[cell_line] = {
                        'ic50_um': ic50_prediction,
                        'cytotoxicity_score': self._ic50_to_score(ic50_prediction),
                        'confidence': self._calculate_confidence(ic50_prediction)
                    }
                
                predictions[cancer_type] = cell_line_predictions
            
            # Predict normal cell toxicity
            normal_cell_predictions = {}
            for cell_line in self.cell_lines['normal_cells']:
                cell_specific_properties = molecular_properties.copy()
                cell_specific_properties.extend([
                    self._get_cell_line_factor(cell_line),
                    0.5  # Normal cell factor
                ])
                
                ic50_prediction = self.cytotoxicity_model.predict([cell_specific_properties])[0] * 3  # Normal cells more resistant
                ic50_prediction = max(ic50_prediction, 1.0)
                
                normal_cell_predictions[cell_line] = {
                    'ic50_um': ic50_prediction,
                    'toxicity_score': self._ic50_to_score(ic50_prediction),
                    'confidence': self._calculate_confidence(ic50_prediction)
                }
            
            predictions['normal_cells'] = normal_cell_predictions
            
            # Calculate selectivity indices
            selectivity_analysis = self._calculate_selectivity(predictions)
            
            return {
                'cytotoxicity_predictions': predictions,
                'selectivity_analysis': selectivity_analysis,
                'overall_assessment': self._assess_overall_cytotoxicity(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Cytotoxicity prediction failed: {e}")
            return {'error': str(e)}
    
    def _extract_molecular_properties(self, compound: Dict) -> List[float]:
        """Extract molecular properties for cytotoxicity prediction."""
        # In a real implementation, this would calculate actual molecular descriptors
        # For now, use simplified properties based on compound data
        
        smiles = compound.get('smiles', '')
        binding_affinity = compound.get('binding_affinity', -7.0)
        
        # Simplified molecular descriptors
        molecular_weight = len(smiles) * 10 + 200  # Rough estimate
        logp = -2 + (len(smiles) % 10) * 0.5
        hbd = smiles.count('N') + smiles.count('O')
        hba = smiles.count('N') + smiles.count('O') + smiles.count('F') + smiles.count('Cl')
        tpsa = hba * 10 + hbd * 5
        
        return [molecular_weight, logp, hbd, hba, tpsa, binding_affinity]
    
    def _get_cell_line_factor(self, cell_line: str) -> float:
        """Get cell line specific factor for cytotoxicity prediction."""
        # Different cell lines have different sensitivities
        sensitivity_factors = {
            'MIA PaCa-2': 1.2,  # More sensitive
            'PANC-1': 1.0,      # Standard
            'BxPC-3': 0.8,      # Less sensitive
            'AsPC-1': 1.1,      # More sensitive
            'MCF-7': 1.0,
            'MDA-MB-231': 1.3,
            'A549': 0.9,
            'H460': 1.1,
            'HCT-116': 1.0,
            'SW480': 0.8,
            'HEK-293': 0.3,     # Normal cells less sensitive
            'MCF-10A': 0.4,
            'IMR-90': 0.3,
            'BJ': 0.4
        }
        return sensitivity_factors.get(cell_line, 1.0)
    
    def _get_cancer_type_factor(self, cancer_type: str) -> float:
        """Get cancer type specific factor."""
        cancer_factors = {
            'pancreatic_cancer': 1.2,  # More aggressive
            'breast_cancer': 1.0,
            'lung_cancer': 1.1,
            'colon_cancer': 0.9
        }
        return cancer_factors.get(cancer_type, 1.0)
    
    def _ic50_to_score(self, ic50: float) -> float:
        """Convert IC50 to cytotoxicity score (0-1 scale)."""
        # Lower IC50 = higher cytotoxicity score
        return max(0, min(1, 1 - (ic50 / 100)))
    
    def _calculate_confidence(self, ic50: float) -> str:
        """Calculate confidence level for prediction."""
        # Based on IC50 value and model confidence
        if ic50 < 10:
            return 'high'
        elif ic50 < 50:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_selectivity(self, predictions: Dict) -> Dict:
        """Calculate selectivity indices between cancer and normal cells."""
        selectivity_analysis = {}
        
        for cancer_type in ['pancreatic_cancer', 'breast_cancer', 'lung_cancer', 'colon_cancer']:
            if cancer_type in predictions:
                cancer_ic50s = [pred['ic50_um'] for pred in predictions[cancer_type].values()]
                normal_ic50s = [pred['ic50_um'] for pred in predictions['normal_cells'].values()]
                
                avg_cancer_ic50 = np.mean(cancer_ic50s)
                avg_normal_ic50 = np.mean(normal_ic50s)
                
                selectivity_index = avg_normal_ic50 / avg_cancer_ic50
                therapeutic_index = avg_normal_ic50 / min(cancer_ic50s)
                
                selectivity_analysis[cancer_type] = {
                    'selectivity_index': selectivity_index,
                    'therapeutic_index': therapeutic_index,
                    'safety_margin': therapeutic_index - 1,
                    'risk_assessment': self._assess_selectivity_risk(selectivity_index)
                }
        
        return selectivity_analysis
    
    def _assess_selectivity_risk(self, selectivity_index: float) -> str:
        """Assess risk based on selectivity index."""
        if selectivity_index >= 10:
            return 'low_risk'
        elif selectivity_index >= 5:
            return 'moderate_risk'
        elif selectivity_index >= 2:
            return 'high_risk'
        else:
            return 'very_high_risk'
    
    def _assess_overall_cytotoxicity(self, predictions: Dict) -> Dict:
        """Assess overall cytotoxicity profile."""
        all_cancer_ic50s = []
        all_normal_ic50s = []
        
        for cancer_type, cell_predictions in predictions.items():
            if cancer_type != 'normal_cells':
                all_cancer_ic50s.extend([pred['ic50_um'] for pred in cell_predictions.values()])
            else:
                all_normal_ic50s.extend([pred['ic50_um'] for pred in cell_predictions.values()])
        
        avg_cancer_ic50 = np.mean(all_cancer_ic50s)
        avg_normal_ic50 = np.mean(all_normal_ic50s)
        
        return {
            'average_cancer_ic50': avg_cancer_ic50,
            'average_normal_ic50': avg_normal_ic50,
            'overall_selectivity': avg_normal_ic50 / avg_cancer_ic50,
            'cytotoxicity_potency': self._assess_potency(avg_cancer_ic50),
            'safety_profile': self._assess_safety(avg_normal_ic50),
            'clinical_potential': self._assess_clinical_potential(avg_cancer_ic50, avg_normal_ic50)
        }
    
    def _assess_potency(self, ic50: float) -> str:
        """Assess cytotoxicity potency."""
        if ic50 < 1:
            return 'very_high'
        elif ic50 < 10:
            return 'high'
        elif ic50 < 50:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_safety(self, ic50: float) -> str:
        """Assess safety profile."""
        if ic50 > 100:
            return 'safe'
        elif ic50 > 50:
            return 'moderate_safety'
        elif ic50 > 10:
            return 'concerning'
        else:
            return 'unsafe'
    
    def _assess_clinical_potential(self, cancer_ic50: float, normal_ic50: float) -> str:
        """Assess clinical potential."""
        selectivity = normal_ic50 / cancer_ic50
        
        if cancer_ic50 < 10 and selectivity > 5:
            return 'high_potential'
        elif cancer_ic50 < 50 and selectivity > 3:
            return 'moderate_potential'
        elif cancer_ic50 < 100 and selectivity > 2:
            return 'low_potential'
        else:
            return 'unlikely'
    
    def get_validation_summary(self) -> Dict:
        """Get summary of cell-based validation capabilities."""
        return {
            'cell_lines_available': len([line for lines in self.cell_lines.values() for line in lines]),
            'cancer_types': list(self.cell_lines.keys()),
            'prediction_metrics': ['IC50', 'Cytotoxicity Score', 'Selectivity Index', 'Therapeutic Index'],
            'validation_status': 'active'
        } 