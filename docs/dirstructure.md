# 📁 Directory Structure + Code Scaffolding for QuantumDock

## 🏗️ Complete Project Structure

```
quantum_dock/
├── app/                          # User interfaces and applications
│   ├── run_app.py               # Streamlit web interface
│   ├── cli_interface.py         # Command-line interface
│   └── app_utils.py             # UI helper functions
│
├── data/                        # Input/output data and configuration
│   ├── cisplatin_context.json   # Ligand substitutions + descriptor weights
│   ├── pancreatic_target.json   # Protein environment modifiers
│   ├── analogs.csv              # Input SMILES or analog metadata
│   ├── docking_scores.csv       # Optional Vina scores for training
│   ├── descriptors.json         # Saved quantum descriptors cache
│   └── training_data.csv        # Training data for QNN model
│
├── molecules/                   # Molecular structure files
│   ├── cisplatin_base.smi       # Base cisplatin SMILES
│   ├── generated_analogs/       # Generated analog structures
│   │   ├── analog_001.xyz       # 3D coordinates
│   │   ├── analog_002.mol       # MOL format structures
│   │   └── ...
│   └── reference_compounds/     # Reference molecular structures
│
├── vqe_engine/                  # Quantum chemistry simulation engine
│   ├── vqe_runner.py            # Main VQE execution and orchestration
│   ├── utils.py                 # Geometry parsing, xyz generation, PySCF setup
│   ├── molecular_descriptors.py # Quantum descriptor calculations
│   └── context_modifiers.py     # Apply biological context to descriptors
│
├── qaoa_optimizer/              # Quantum pose optimization (optional)
│   ├── qaoa_docking.py          # QAOA Hamiltonian + cost optimizer
│   ├── pose_encoding.py         # Molecular pose to binary encoding
│   └── hamiltonian_builder.py   # Cost function construction
│
├── qnn_model/                   # Quantum Neural Network components
│   ├── qnn_predictor.py         # Angle-encoded QNN for affinity prediction
│   ├── train_qnn.py             # Training script for QNN using labels
│   ├── qnn_utils.py             # QNN evaluation and feature handling
│   ├── multi_task_qnn.py        # Multi-task QNN for toxicity/resistance
│   └── model_checkpoints/       # Saved trained models
│
├── agent_core/                  # Core agent logic and orchestration
│   ├── data_loader.py           # Load and parse configuration files
│   ├── analog_generator.py      # Generate molecular analogs
│   ├── scoring_engine.py        # Final scoring and ranking logic
│   ├── simple_agent.py          # Rule-based analog mutator and loop
│   ├── rl_agent.py              # Reinforcement learning agent (future)
│   └── pipeline_orchestrator.py # Main pipeline coordination
│
├── results/                     # Output files and analysis
│   ├── top_candidates.csv       # Ranked analog candidates
│   ├── detailed_scores.json     # Detailed scoring breakdown
│   ├── vqe_results/            # VQE computation results
│   ├── qnn_predictions/        # QNN prediction outputs
│   └── performance_metrics.json # Pipeline performance data
│
├── tests/                       # Test suite
│   ├── test_vqe_engine.py      # VQE engine tests
│   ├── test_qnn_model.py       # QNN model tests
│   ├── test_agent_core.py      # Agent core functionality tests
│   └── test_data/              # Test data files
│
├── configs/                     # Configuration files
│   ├── vqe_config.yaml         # VQE simulation parameters
│   ├── qnn_config.yaml         # QNN architecture parameters
│   └── pipeline_config.yaml    # Pipeline execution parameters
│
├── docs/                        # Documentation
│   ├── objective.md             # Project objectives and scope
│   ├── finalmvp.md             # MVP implementation instructions
│   ├── dirstructure.md         # This file - directory structure
│   └── api_docs.md             # API documentation
│
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation script
├── README.md                   # Project overview and quick start
├── main.py                     # Main entrypoint: build pipeline, run inference
└── .gitignore                  # Git ignore patterns
```

---

## 📋 File Specifications and Requirements

### 🔧 Core Implementation Files

#### `main.py` - Main Entry Point
**Purpose:** Pipeline orchestration and execution
**Key Functions:**
- `main()` - Entry point for full pipeline
- `run_inference_mode()` - Run inference on new analogs
- `run_training_mode()` - Train QNN models
- `setup_logging()` - Configure logging system

#### `agent_core/data_loader.py` - Configuration Loader
**Purpose:** Load and validate configuration files
**Required Functions:**
```python
def load_cisplatin_context(file_path: str) -> dict
def load_pancreatic_target(file_path: str) -> dict
def validate_config(config: dict) -> bool
```

#### `agent_core/analog_generator.py` - Molecular Generation
**Purpose:** Generate cisplatin analogs with systematic substitutions
**Required Functions:**
```python
def generate_analogs(base_smiles: str, substitutions: dict) -> list
def smiles_to_xyz(smiles: str) -> str
def validate_molecule(mol_structure) -> bool
```

#### `vqe_engine/vqe_runner.py` - VQE Simulation
**Purpose:** Quantum chemistry computations
**Required Functions:**
```python
def run_vqe_descriptors(xyz_path: str) -> dict
def calculate_homo_lumo_gap(mol) -> float
def calculate_dipole_moment(mol) -> float
def apply_context_modifiers(descriptors: dict, context: dict) -> dict
```

#### `qnn_model/qnn_predictor.py` - QNN Model
**Purpose:** Quantum neural network for binding affinity prediction
**Required Classes:**
```python
class QNNPredictor:
    def __init__(self, n_features: int, n_layers: int)
    def create_circuit(self, features: list, weights: list)
    def train(self, training_data: list, labels: list)
    def predict(self, features: list) -> float
```

#### `agent_core/scoring_engine.py` - Scoring System
**Purpose:** Calculate final scores and rank analogs
**Required Functions:**
```python
def calculate_final_score(binding_affinity: float, resistance: float, toxicity: float) -> float
def rank_analogs(analogs_with_scores: list) -> list
def save_results(ranked_analogs: list, output_path: str)
```

---

### 📊 Data Files Specifications

#### `data/cisplatin_context.json` - Ligand Context
**Structure:**
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

#### `data/pancreatic_target.json` - Target Environment
**Structure:**
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

#### `data/analogs.csv` - Input Analogs
**Required Columns:**
- `analog_id`: Unique identifier
- `smiles`: SMILES string
- `substitution_type`: Type of modification
- `source`: Generation method

---

### ⚙️ Configuration Files

#### `configs/vqe_config.yaml` - VQE Parameters
```yaml
basis_set: "6-31G(d,p)"
pt_basis_set: "LANL2DZ"
optimization_method: "L-BFGS-B"
convergence_threshold: 1e-6
max_iterations: 1000
```

#### `configs/qnn_config.yaml` - QNN Architecture
```yaml
n_features: 3
n_layers: 2
n_qubits: 6
learning_rate: 0.01
batch_size: 32
epochs: 100
```

#### `configs/pipeline_config.yaml` - Pipeline Settings
```yaml
max_analogs: 200
parallel_vqe: true
save_intermediate: true
output_format: "csv"
log_level: "INFO"
```

---

### 🧪 Testing Structure

#### `tests/test_vqe_engine.py` - VQE Tests
**Test Coverage:**
- VQE descriptor calculation accuracy
- Context modifier application
- Error handling for invalid molecules

#### `tests/test_qnn_model.py` - QNN Tests
**Test Coverage:**
- QNN circuit construction
- Training convergence
- Prediction accuracy

#### `tests/test_agent_core.py` - Agent Tests
**Test Coverage:**
- Analog generation functionality
- Scoring engine accuracy
- Pipeline orchestration

---

### 📱 User Interface Files

#### `app/run_app.py` - Streamlit Interface
**Features:**
- Upload molecular structures
- Configure simulation parameters
- Visualize results and rankings
- Download output files

#### `app/cli_interface.py` - Command Line Interface
**Commands:**
- `generate` - Generate new analogs
- `simulate` - Run VQE simulations
- `train` - Train QNN models
- `predict` - Run predictions on analogs

---

### 🔍 Output Files

#### `results/top_candidates.csv` - Ranked Results
**Columns:**
- `analog_id`, `smiles`, `final_score`, `binding_affinity`
- `resistance_score`, `toxicity_score`, `energy`, `homo_lumo_gap`
- `dipole_moment`, `rank`

#### `results/detailed_scores.json` - Detailed Analysis
**Structure:**
```json
{
  "analog_id": "analog_001",
  "scores": {
    "vqe_descriptors": {...},
    "qnn_prediction": 0.85,
    "final_score": 0.73
  },
  "metadata": {...}
}
```

---

## 🚀 Implementation Priority

### Phase 1: Core Pipeline
1. `main.py` - Basic pipeline orchestration
2. `agent_core/data_loader.py` - Configuration loading
3. `agent_core/analog_generator.py` - Molecular generation
4. `vqe_engine/vqe_runner.py` - VQE simulation
5. `qnn_model/qnn_predictor.py` - QNN implementation
6. `agent_core/scoring_engine.py` - Scoring and ranking

### Phase 2: Enhancement
1. `app/run_app.py` - Streamlit interface
2. `qaoa_optimizer/qaoa_docking.py` - QAOA optimization
3. `qnn_model/multi_task_qnn.py` - Multi-task learning
4. `agent_core/rl_agent.py` - Reinforcement learning

### Phase 3: Production
1. Comprehensive testing suite
2. Performance optimization
3. Documentation and API
4. Database integration
