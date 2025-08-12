# 🧬 MoleQule Benchmarking System

This directory contains the comprehensive benchmarking system for validating MoleQule's molecular docking accuracy against industry standards.

## 📋 Overview

The benchmarking system follows the protocol outlined in `docs/benchmark.md` and consists of three main phases:

1. **Phase 1: Dataset Curation** - Curate cisplatin analog dataset with experimental data
2. **Phase 2: Docking Method Validation** - Validate all MoleQule docking methods
3. **Phase 3: Comparative Benchmarking** - Compare against industry standards

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run Complete Benchmark

```bash
python run_benchmark.py
```

### Run Specific Phases

```bash
# Run only dataset curation
python run_benchmark.py --phases phase_1

# Run docking validation and comparative benchmarking
python run_benchmark.py --phases phase_2 phase_3
```

## 📁 Directory Structure

```
benchmarking/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── run_benchmark.py           # Main benchmark runner
├── README.md                  # This file
├── data/                      # Generated datasets
│   ├── cisplatin_analog_dataset.csv
│   ├── target_proteins_dataset.csv
│   ├── molecular_validation_results.json
│   └── dataset_summary.json
├── results/                   # Benchmark results
│   ├── docking_validation_results.json
│   ├── docking_validation_summary.csv
│   ├── comparative_benchmark_results.json
│   ├── comparative_benchmark_data.csv
│   ├── performance_rankings.json
│   └── molequle_benchmark_YYYYMMDD_HHMMSS_results.json
└── scripts/                   # Phase-specific scripts
    ├── 01_dataset_curation.py
    ├── 02_docking_validation.py
    └── 03_comparative_benchmarking.py
```

## 🔬 Phase Details

### Phase 1: Dataset Curation

**Script**: `scripts/01_dataset_curation.py`

**Purpose**: Curate a comprehensive dataset of cisplatin analogs with experimental binding data.

**Outputs**:
- `data/cisplatin_analog_dataset.csv` - Main compound-target dataset
- `data/target_proteins_dataset.csv` - Target protein information
- `data/molecular_validation_results.json` - Molecular structure validation
- `data/dataset_summary.json` - Dataset statistics

**Key Features**:
- 30+ cisplatin analogs with experimental IC50 data
- Multiple target proteins (DNA, GSTP1, KRAS, TP53)
- Molecular structure validation using RDKit
- Quality control and provenance tracking

### Phase 2: Docking Method Validation

**Script**: `scripts/02_docking_validation.py`

**Purpose**: Validate all MoleQule docking methods against the curated dataset.

**Methods Tested**:
- **Basic Analysis**: Simple geometric analysis
- **QAOA Quantum**: Quantum-enhanced pose optimization
- **Classical Force Field**: RDKit UFF force field docking
- **Grid Search**: Systematic conformational sampling
- **AutoDock Vina**: Industry standard integration

**Outputs**:
- `results/docking_validation_results.json` - Detailed validation results
- `results/docking_validation_summary.csv` - Performance summary

**Metrics Calculated**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of determination)
- Spearman correlation
- Pose accuracy metrics

### Phase 3: Comparative Benchmarking

**Script**: `scripts/03_comparative_benchmarking.py`

**Purpose**: Compare MoleQule performance against industry standards.

**Baseline Methods**:
- AutoDock Vina (industry standard)
- Glide (Schrödinger commercial)
- GOLD (Cambridge Crystallographic Data Centre)
- FlexX (BioSolveIT)

**Outputs**:
- `results/comparative_benchmark_results.json` - Comprehensive comparison
- `results/comparative_benchmark_data.csv` - Comparison dataset
- `results/performance_rankings.json` - Method rankings

**Statistical Analysis**:
- Paired t-tests
- Effect size calculations (Cohen's d)
- Confidence intervals
- Multiple testing corrections

## 📊 Success Criteria

The benchmarking system validates against the following success criteria from `benchmark.md`:

### Technical Performance
- **RMSE Target**: <2.0 kcal/mol (vs AutoDock Vina baseline)
- **Statistical Significance**: p < 0.05 for improvements
- **Reproducibility**: 95% confidence interval overlap

### Experimental Validation
- **Literature Correlation**: R² > 0.75 with experimental data
- **Pose Accuracy**: >70% correct binding mode prediction
- **Rank Order Accuracy**: >80% correct potency ordering

### Commercial Readiness
- **Performance Claims**: Validated by independent testing
- **Competitive Advantage**: Demonstrable superiority
- **Market Validation**: Customer pilot feedback

## ⚙️ Configuration

The benchmarking system is configured via `config.yaml`:

```yaml
# Dataset Configuration
dataset:
  cisplatin_analogs:
    min_compounds: 30
    data_sources: ["PDBbind", "BindingDB", "ChEMBL", "PubChem"]

# Docking Methods
docking_methods:
  molequle_methods:
    - name: "basic_analysis"
    - name: "qaoa_quantum"
    - name: "classical_force_field"
    - name: "grid_search"
    - name: "autodock_vina"

# Success Criteria
success_criteria:
  technical_performance:
    min_rmse: 2.0
    statistical_significance: 0.05
```

## 📈 Results Interpretation

### Performance Rankings

The system generates rankings across multiple metrics:

1. **RMSE Ranking** (Lower is better)
2. **R² Ranking** (Higher is better)
3. **Pose Accuracy Ranking** (Higher is better)
4. **Overall Ranking** (Composite score)

### Statistical Significance

Results include:
- P-values for hypothesis testing
- Effect sizes (Cohen's d)
- Confidence intervals
- Multiple testing corrections

### Success Assessment

The system automatically assesses:
- Technical performance targets
- Experimental validation criteria
- Commercial readiness indicators

## 🔧 Customization

### Adding New Docking Methods

1. Add method to `config.yaml`
2. Implement method in `scripts/02_docking_validation.py`
3. Update validation pipeline

### Modifying Success Criteria

1. Edit thresholds in `config.yaml`
2. Update assessment logic in scripts
3. Re-run benchmarking

### Extending Dataset

1. Add compounds to `scripts/01_dataset_curation.py`
2. Include experimental data sources
3. Update quality control pipeline

## 📝 Reporting

### Generated Reports

The system generates comprehensive reports including:
- Phase-specific validation reports
- Comparative analysis reports
- Statistical significance reports
- Success criteria assessment

### File Formats

- **JSON**: Detailed results and metadata
- **CSV**: Tabular data for analysis
- **Console**: Real-time progress and summary

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **RDKit Installation Issues**
   ```bash
   conda install -c conda-forge rdkit
   ```

3. **Memory Issues**
   - Reduce sample size in configuration
   - Use subset of compounds for testing

4. **File Permission Errors**
   - Ensure write permissions to `data/` and `results/` directories

### Debug Mode

Run individual phases for debugging:

```bash
python scripts/01_dataset_curation.py
python scripts/02_docking_validation.py
python scripts/03_comparative_benchmarking.py
```

## 📚 References

- **Benchmark Protocol**: `docs/benchmark.md`
- **Configuration**: `config.yaml`
- **Results**: `results/` directory
- **Datasets**: `data/` directory

## 🤝 Contributing

To contribute to the benchmarking system:

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include statistical validation
4. Update documentation
5. Test with sample data

## 📄 License

This benchmarking system is part of the MoleQule project and follows the same licensing terms. 