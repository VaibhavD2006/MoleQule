# 🚀 Final MVP Instructions: QuantumDock — A Quantum Agent Replacement for AutoDock Vina

## 🎯 Goal
Design a **biologically intelligent quantum docking agent** that replaces AutoDock Vina using quantum simulations, QNN-based prediction, and biological context awareness to identify optimized cisplatin analogs for pancreatic cancer treatment.

---

## 🧠 System Overview

| Component             | Function                                                             |
|-----------------------|----------------------------------------------------------------------|
| Input Handler         | Loads SMILES or `.mol`, generates 3D `.xyz` structure                |
| Quantum Simulator (VQE) | Extracts quantum descriptors (energy, gap, dipole)                |
| Pose Optimizer (QAOA) (Optional) | Searches for optimal binding configuration            |
| Context Modulator     | Reads cisplatin + pancreatic target JSON files and modifies weights |
| QNN Model             | Predicts binding affinity (with resistance/efficacy awareness)       |
| Ranking Engine        | Sorts analogs by effectiveness                                       |
| Agent Loop            | Iteratively mutates and evaluates analogs                            |

---

## 📋 Prerequisites and Dependencies

### Required Python Packages
```txt
pennylane>=0.32.0
rdkit>=2023.03.1
pyscf>=2.1.1
numpy>=1.24.0
pandas>=2.0.0
streamlit>=1.25.0
scikit-learn>=1.3.0
qiskit>=0.44.0 (optional)
```

### Required Files Structure
```
data/
├── cisplatin_context.json    # Ligand substitutions + descriptor weights
├── pancreatic_target.json    # Protein environment modifier values
└── analogs.csv              # Input SMILES or analog metadata
```

---

## ✅ MVP Build Instructions

### 🔷 Step 1: Setup Input Data Files

**File: `data/cisplatin_context.json`**
```json
{
  "base_smiles": "N[Pt](N)(Cl)Cl",
  "ligand_substitutions": {
    "ammine": ["methyl", "ethyl", "propyl"],
    "chloride": ["bromide", "iodide", "acetate"]
  },
  "descriptor_weights": {
    "energy_weight": 0.4,
    "gap_weight": 0.3,
    "dipole_weight": 0.3
  }
}
```

**File: `data/pancreatic_target.json`**
```json
{
  "environment_modifiers": {
    "ph_modifier": 0.85,
    "hypoxia_modifier": 0.9,
    "stromal_barrier_modifier": 0.7
  },
  "resistance_factors": {
    "gstp1_weight": 0.8,
    "efflux_pump_weight": 0.6,
    "dna_repair_weight": 0.7
  }
}
```

**Implementation Location:** `agent_core/data_loader.py`

---

### ⚛️ Step 2: Generate & Convert Analogs

**Function: `generate_analogs()`**
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import json

def generate_analogs(base_smiles, context_file):
    """Generate cisplatin analogs based on context file substitutions"""
    mol = Chem.MolFromSmiles(base_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Apply ligand substitutions
    # Generate 3D .xyz structure
    # Return list of analog structures
    return analogs
```

**Implementation Location:** `agent_core/analog_generator.py`

---

### ⚛️ Step 3: Quantum Descriptor Simulation (VQE)

**Function: `run_vqe_descriptors()`**
```python
import pennylane as qml
from pyscf import gto, scf

def run_vqe_descriptors(xyz_path):
    """Run VQE simulation to compute quantum descriptors"""
    
    # Compute:
    # - Ground-state energy
    # - HOMO-LUMO gap  
    # - Dipole moment
    
    descriptors = {
        'energy': energy_value,
        'homo_lumo_gap': gap_value,
        'dipole_moment': dipole_value
    }
    
    # Apply context modifiers
    modified_descriptors = apply_context_modifiers(descriptors)
    
    return modified_descriptors
```

**Formula for Context Application:**
```python
def apply_context_modifiers(descriptors, context, target_info):
    """Apply biological context modifiers to quantum descriptors"""
    modified_energy = (descriptors['energy'] * 
                      target_info['environment_modifiers']['ph_modifier'] * 
                      context['descriptor_weights']['energy_weight'])
    
    # Apply similar modifications to other descriptors
    return modified_descriptors
```

**Implementation Location:** `vqe_engine/vqe_runner.py`

---

### 🧩 Step 4: (Optional) QAOA Pose Optimization

**Function: `qaoa_pose_optimization()`**
```python
import pennylane as qml
from pennylane import qaoa

def qaoa_pose_optimization(molecular_config):
    """Use QAOA to find optimal binding pose"""
    
    # Convert molecular configuration to binary pose encoding
    # Define cost Hamiltonian: H_cost = binding_energy(pose)
    # Run QAOA optimization
    
    optimal_pose = qaoa.optimize(cost_hamiltonian, steps=10)
    
    return optimal_pose
```

**Implementation Location:** `qaoa_optimizer/qaoa_docking.py`

---

### 🧠 Step 5: QNN Binding Affinity Prediction

**QNN Architecture:**
```python
import pennylane as qml

def create_qnn_model(n_features=3, n_layers=2):
    """Create QNN model for binding affinity prediction"""
    
    dev = qml.device("default.qubit", wires=n_features)
    
    @qml.qnode(dev)
    def qnn_circuit(features, weights):
        # Angle encoding
        for i in range(n_features):
            qml.RY(features[i], wires=i)
        
        # Entanglement layers
        for layer in range(n_layers):
            for i in range(n_features):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_features - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    return qnn_circuit
```

**Training Function:**
```python
def train_qnn(qnn_model, training_data, labels):
    """Train QNN on binding affinity data"""
    
    # Training data sources:
    # - Vina scores (if surrogate mode)
    # - Experimental IC50 / binding affinity
    
    # Optimize QNN parameters
    # Return trained model
    
    return trained_model
```

**Implementation Location:** `qnn_model/qnn_predictor.py`

---

### 🧪 Step 6: Analog Scoring and Ranking

**Scoring Function:**
```python
def calculate_final_score(binding_affinity, resistance_score, toxicity_score):
    """Calculate final analog score"""
    
    final_score = (binding_affinity - 
                  resistance_score * 0.3 - 
                  toxicity_score * 0.2)
    
    return final_score
```

**Ranking Function:**
```python
def rank_analogs(analogs_with_scores):
    """Sort analogs by effectiveness criteria"""
    
    # Sort by:
    # 1. Highest predicted binding affinity
    # 2. Lowest predicted resistance
    # 3. Novel ligand structure (diversity bonus)
    
    ranked_analogs = sorted(analogs_with_scores, key=lambda x: x['final_score'], reverse=True)
    
    # Save to results/top_candidates.csv
    return ranked_analogs
```

**Implementation Location:** `agent_core/scoring_engine.py`

---

## 🔁 Surrogate Learning Mode

**Training Pipeline:**
```python
def surrogate_training_pipeline():
    """Train QNN using existing Vina scores"""
    
    # Input: VQE descriptors
    # Output: Vina binding affinity
    # Fine-tune: on biological data (IC50, mutation-specific responses)
    
    return surrogate_model
```

**Implementation Location:** `qnn_model/train_qnn.py`

---

## 📈 QuantumDock vs AutoDock Vina Comparison

| Feature | AutoDock Vina | QuantumDock Agent |
|---------|---------------|-------------------|
| Pose search | Random sampling | QAOA-guided (optional) |
| Scoring function | Hardcoded physics | Adaptive QNN (learned from data) |
| Descriptor type | Force-field | Quantum descriptors (VQE) |
| Adaptability | None | Agent-based tuning |
| Target-specificity | None | Biologically contextual |
| Evolvability | None | Agent learns over time |

---

## ✅ MVP Implementation Checklist

| Task | Implementation File | Status |
|------|-------------------|---------|
| Load biological context from JSON | `agent_core/data_loader.py` | ⏳ |
| Generate and convert analogs | `agent_core/analog_generator.py` | ⏳ |
| Run VQE to get descriptors | `vqe_engine/vqe_runner.py` | ⏳ |
| Apply protein/disease-based weighting | `vqe_engine/utils.py` | ⏳ |
| Predict binding with QNN | `qnn_model/qnn_predictor.py` | ⏳ |
| Score, rank, and save analogs | `agent_core/scoring_engine.py` | ⏳ |
| Main pipeline orchestration | `main.py` | ⏳ |

---

## 🧠 Optional Next Steps

| Feature | Benefit | Implementation File |
|---------|---------|-------------------|
| Reinforcement Agent | Learns best substitutions automatically | `agent_core/rl_agent.py` |
| Streamlit UI | Interactive demo interface | `app/run_app.py` |
| Database backend | Persistent analog storage | `data/database.py` |
| Multi-task QNN | Predict toxicity, resistance, selectivity | `qnn_model/multi_task_qnn.py` |

---

## 🔧 Technical Specifications

### Quantum Circuit Parameters
- **Qubits:** 3-6 (depending on feature count)
- **Circuit Depth:** 2-4 layers
- **Encoding:** Angle encoding with RY gates
- **Entanglement:** Linear topology with CNOT gates

### VQE Configuration
- **Basis Sets:** 6-31G(d,p) for light atoms, LANL2DZ for Pt
- **Optimization:** L-BFGS-B or COBYLA
- **Convergence:** 1e-6 energy tolerance

### Performance Targets
- **Analog Generation:** 50-200 analogs per run
- **VQE Simulation:** <5 minutes per molecule
- **QNN Training:** <30 minutes for 1000 samples
- **End-to-end Pipeline:** <2 hours for full analog set