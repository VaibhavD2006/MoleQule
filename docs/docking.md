# Molecular Docking and Visualization Integration for MoleQule

## Overview
This document outlines the implementation of molecular docking visualization features for the MoleQule platform. These features will allow users to visualize how their generated analogs bind to biological targets.

## Table of Contents
1. [Architecture Overview](#architecture)
2. [Component Implementation](#components)
3. [Binding Site Detection](#binding-sites)
4. [3D Visualization](#visualization)
5. [Docking Integration](#docking)
6. [Frontend Integration](#frontend)
7. [Dependencies](#dependencies)
8. [Implementation Steps](#steps)

---

## Architecture Overview {#architecture}

### System Components
```
Frontend (React/Next.js)
    ↓
Backend API (FastAPI)
    ↓
Docking Service (Python)
    ↓
Visualization Engine (RDKit + Py3Dmol)
```

### Data Flow
1. User uploads molecule → Generate analogs
2. Select target protein (DNA, GSTP1, etc.)
3. Run docking simulation
4. Generate 3D binding pose visualization
5. Display interactive results

---

## Component Implementation {#components}

### 1. Docking Service Backend

**File:** `molequle/docking_service/main.py`
```python
from fastapi import FastAPI, HTTPException
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from Bio.PDB import PDBParser

app = FastAPI(title="MoleQule Docking Service")

@app.post("/detect-binding-sites")
async def detect_binding_sites(target_pdb: str):
    """Detect binding sites in target protein"""
    # Implementation details below

@app.post("/dock-molecule")
async def dock_molecule(ligand_mol: str, target_pdb: str):
    """Perform molecular docking"""
    # Implementation details below

@app.post("/visualize-complex")
async def visualize_complex(ligand_mol: str, target_pdb: str, pose_data: dict):
    """Generate 3D visualization of binding complex"""
    # Implementation details below
```

### 2. Frontend Visualization Component

**File:** `molequle/frontend/src/components/DockingVisualization.tsx`
```typescript
import React, { useEffect, useRef } from 'react';

interface DockingVisualizationProps {
  ligandData: string;
  targetData: string;
  poseData?: any;
}

export const DockingVisualization: React.FC<DockingVisualizationProps> = ({
  ligandData,
  targetData,
  poseData
}) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  
  // Implementation details below
};
```

---

## Binding Site Detection {#binding-sites}

### Method 1: PDB-based Automatic Detection

**Implementation:** `molequle/docking_service/binding_sites.py`
```python
from Bio.PDB import PDBParser
import numpy as np
from typing import List, Dict, Tuple

class BindingSiteDetector:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
    
    def detect_cavities(self, pdb_file: str) -> List[Dict]:
        """
        Detect binding cavities in protein structure
        
        Args:
            pdb_file (str): Path to PDB file
            
        Returns:
            List[Dict]: List of detected binding sites with coordinates
        """
        structure = self.parser.get_structure("target", pdb_file)
        
        # Algorithm implementation
        binding_sites = []
        
        for model in structure:
            for chain in model:
                # Identify potential binding pockets
                cavity_coords = self._find_cavities(chain)
                
                for i, coords in enumerate(cavity_coords):
                    binding_sites.append({
                        "site_id": f"site_{i}",
                        "center": coords["center"],
                        "radius": coords["radius"],
                        "score": coords["score"]
                    })
        
        return binding_sites
    
    def _find_cavities(self, chain) -> List[Dict]:
        """Find cavities using geometric analysis"""
        # Implementation for cavity detection
        # Using alpha shapes or grid-based methods
        pass
```

### Method 2: Predefined Binding Regions

**Implementation:** `molequle/docking_service/predefined_sites.py`
```python
# Known binding sites for common targets
PREDEFINED_SITES = {
    "DNA_guanine_N7": {
        "description": "Cisplatin binding site at N7 of guanine",
        "coordinates": [10.5, 15.2, 8.7],
        "radius": 3.0,
        "target_atoms": ["N7"]
    },
    "GSTP1_active_site": {
        "description": "GSTP1 enzyme active site",
        "coordinates": [25.1, 30.4, 12.8],
        "radius": 4.5,
        "target_atoms": ["CYS47", "SER51"]
    }
}

def get_predefined_site(target_name: str) -> Dict:
    """Get predefined binding site coordinates"""
    return PREDEFINED_SITES.get(target_name, None)
```

---

## 3D Visualization {#visualization}

### Basic Ligand Visualization

**Implementation:** `molequle/docking_service/visualizer.py`
```python
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem

class MolecularVisualizer:
    
    def visualize_ligand(self, mol_file: str, width: int = 500, height: int = 400) -> str:
        """
        Create 3D visualization of ligand molecule
        
        Args:
            mol_file (str): Path to MOL file
            width, height (int): Viewer dimensions
            
        Returns:
            str: HTML string for 3D viewer
        """
        # Load molecule
        ligand = Chem.MolFromMolFile(mol_file)
        if not ligand:
            raise ValueError("Invalid molecule file")
        
        # Generate 3D coordinates
        ligand = Chem.AddHs(ligand)
        AllChem.EmbedMolecule(ligand)
        AllChem.UFFOptimizeMolecule(ligand)
        
        # Create 3D viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(open(mol_file).read(), 'mol')
        view.setStyle({'stick': {'colorscheme': 'element'}})
        view.zoomTo()
        
        return view._make_html()
    
    def visualize_complex(self, ligand_file: str, receptor_file: str) -> str:
        """
        Visualize ligand-receptor complex
        
        Args:
            ligand_file (str): Path to ligand MOL file
            receptor_file (str): Path to receptor PDB file
            
        Returns:
            str: HTML string for complex viewer
        """
        view = py3Dmol.view(width=600, height=500)
        
        # Add receptor (protein/DNA)
        view.addModel(open(receptor_file).read(), "pdb")
        view.setStyle({'model': 0}, {"cartoon": {'color': 'lightblue'}})
        
        # Add ligand
        view.addModel(open(ligand_file).read(), "mol")
        view.setStyle({'model': 1}, {"stick": {'colorscheme': 'element'}})
        
        # Center view
        view.zoomTo()
        
        return view._make_html()
```

### Interactive Binding Site Highlighting

```python
def highlight_binding_interactions(self, complex_data: Dict) -> str:
    """
    Highlight specific binding interactions
    
    Args:
        complex_data (Dict): Complex with interaction data
        
    Returns:
        str: HTML with highlighted interactions
    """
    view = py3Dmol.view(width=700, height=600)
    
    # Add models
    view.addModel(complex_data['receptor'], "pdb")
    view.addModel(complex_data['ligand'], "mol")
    
    # Style receptor
    view.setStyle({'model': 0}, {"cartoon": {'color': 'white'}})
    
    # Style ligand
    view.setStyle({'model': 1}, {"stick": {'colorscheme': 'element'}})
    
    # Highlight binding residues
    binding_residues = complex_data.get('binding_residues', [])
    for residue in binding_residues:
        view.addStyle(
            {'model': 0, 'resi': residue},
            {'stick': {'color': 'red', 'radius': 0.3}}
        )
    
    # Add distance labels for hydrogen bonds
    h_bonds = complex_data.get('hydrogen_bonds', [])
    for bond in h_bonds:
        view.addLabel(
            f"{bond['distance']:.1f}Å",
            {'position': bond['midpoint'], 'backgroundColor': 'yellow'}
        )
    
    return view._make_html()
```

---

## Docking Integration {#docking}

### QAOA-based Pose Optimization

**Implementation:** `molequle/docking_service/qaoa_docking.py`
```python
from typing import Dict, List
import numpy as np

class QAOAPoseOptimizer:
    """Quantum-enhanced molecular docking using QAOA"""
    
    def optimize_pose(self, ligand: str, binding_site: Dict) -> Dict:
        """
        Optimize ligand pose using QAOA
        
        Args:
            ligand (str): SMILES or MOL string
            binding_site (Dict): Binding site coordinates and properties
            
        Returns:
            Dict: Optimized pose with energy and coordinates
        """
        # Convert ligand to 3D coordinates
        coords = self._generate_conformers(ligand)
        
        # Define optimization problem
        energy_landscape = self._build_energy_function(coords, binding_site)
        
        # Run QAOA optimization
        optimal_pose = self._run_qaoa_optimization(energy_landscape)
        
        return {
            "pose_coordinates": optimal_pose["coordinates"],
            "binding_energy": optimal_pose["energy"],
            "optimization_steps": optimal_pose["steps"],
            "quantum_solution": True
        }
    
    def _generate_conformers(self, ligand: str) -> np.ndarray:
        """Generate multiple conformers for ligand"""
        # RDKit conformer generation
        pass
    
    def _build_energy_function(self, coords: np.ndarray, site: Dict) -> callable:
        """Build energy function for optimization"""
        # Implement energy function based on:
        # - Van der Waals interactions
        # - Electrostatic interactions  
        # - Hydrogen bonding
        # - Steric clashes
        pass
    
    def _run_qaoa_optimization(self, energy_func: callable) -> Dict:
        """Run QAOA to find optimal binding pose"""
        # Quantum optimization implementation
        pass
```

### Classical Docking Fallback

```python
import subprocess
from pathlib import Path

class ClassicalDocker:
    """Classical docking using AutoDock Vina or similar"""
    
    def dock_with_vina(self, ligand_file: str, receptor_file: str, 
                      binding_site: Dict) -> Dict:
        """
        Perform docking using AutoDock Vina
        
        Args:
            ligand_file (str): Path to ligand file
            receptor_file (str): Path to receptor file
            binding_site (Dict): Binding site definition
            
        Returns:
            Dict: Docking results with poses and scores
        """
        # Prepare configuration file
        config = self._create_vina_config(binding_site)
        
        # Run AutoDock Vina
        cmd = [
            "vina",
            "--ligand", ligand_file,
            "--receptor", receptor_file,
            "--config", config,
            "--out", "docked_poses.pdbqt"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Vina docking failed: {result.stderr}")
        
        # Parse results
        poses = self._parse_vina_output("docked_poses.pdbqt")
        
        return {
            "poses": poses,
            "best_score": poses[0]["score"] if poses else None,
            "method": "autodock_vina"
        }
```

---

## Frontend Integration {#frontend}

### React Component for 3D Visualization

**Implementation:** `molequle/frontend/src/components/DockingVisualization.tsx`
```typescript
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

interface DockingResults {
  poses: Array<{
    coordinates: number[][];
    score: number;
    interactions: any[];
  }>;
  visualization_html: string;
}

export const DockingVisualization: React.FC<{
  analogId: string;
  targetProtein: string;
}> = ({ analogId, targetProtein }) => {
  const [dockingResults, setDockingResults] = useState<DockingResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPose, setSelectedPose] = useState(0);
  const viewerRef = useRef<HTMLDivElement>(null);

  const runDocking = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/v1/dock-molecule', {
        analog_id: analogId,
        target_protein: targetProtein,
        method: 'qaoa'  // or 'classical'
      });
      
      setDockingResults(response.data);
    } catch (error) {
      console.error('Docking failed:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (dockingResults && viewerRef.current) {
      // Inject 3D visualization HTML
      viewerRef.current.innerHTML = dockingResults.visualization_html;
    }
  }, [dockingResults, selectedPose]);

  return (
    <div className="docking-visualization">
      <div className="controls">
        <button 
          onClick={runDocking} 
          disabled={loading}
          className="btn btn-primary"
        >
          {loading ? 'Running Docking...' : 'Run Molecular Docking'}
        </button>
        
        {dockingResults && (
          <div className="pose-selector">
            <label>Binding Pose:</label>
            <select 
              value={selectedPose} 
              onChange={(e) => setSelectedPose(Number(e.target.value))}
            >
              {dockingResults.poses.map((pose, index) => (
                <option key={index} value={index}>
                  Pose {index + 1} (Score: {pose.score.toFixed(2)})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="viewer-container">
        <div 
          ref={viewerRef} 
          className="molecular-viewer"
          style={{ width: '100%', height: '500px' }}
        />
        
        {dockingResults && (
          <div className="binding-info">
            <h4>Binding Analysis</h4>
            <p>Best Score: {dockingResults.poses[0]?.score.toFixed(2)} kcal/mol</p>
            <div className="interactions">
              {/* Display binding interactions */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
```

### Results Page Integration

**Add to:** `molequle/frontend/src/pages/results/[jobId].js`
```javascript
import { DockingVisualization } from '../../components/DockingVisualization';

// Add to existing results page
const DockingSection = ({ analogs }) => (
  <div className="docking-section">
    <h3>Molecular Docking Analysis</h3>
    <div className="target-selection">
      <label>Select Target:</label>
      <select>
        <option value="DNA">DNA (Guanine N7)</option>
        <option value="GSTP1">GSTP1 Enzyme</option>
        <option value="custom">Upload Custom Target</option>
      </select>
    </div>
    
    {analogs.map((analog) => (
      <div key={analog.id} className="analog-docking">
        <h4>{analog.analog_id}</h4>
        <DockingVisualization 
          analogId={analog.id}
          targetProtein="DNA" 
        />
      </div>
    ))}
  </div>
);
```

---

## Dependencies {#dependencies}

### Python Dependencies
Add to `molequle/docking_service/requirements.txt`:
```
fastapi
uvicorn
rdkit
py3Dmol
biopython
numpy
scipy
openbabel-python
autodock-vina  # optional
pennylane      # for QAOA
```

### Frontend Dependencies
Add to `molequle/frontend/package.json`:
```json
{
  "dependencies": {
    "3dmol": "^1.8.0",
    "react-3dmol": "^1.0.0"
  }
}
```

---

## Implementation Steps {#steps}

### Phase 1: Basic Visualization (Week 1)
1. **Setup docking service structure**
   - Create `molequle/docking_service/` directory
   - Implement basic FastAPI service
   - Add molecular visualization with RDKit + Py3Dmol

2. **Frontend integration**
   - Create `DockingVisualization` component
   - Add basic 3D molecule viewer
   - Integrate with existing results page

3. **Test with sample molecules**
   - Test visualization with cisplatin analogs
   - Verify 3D rendering works correctly

### Phase 2: Binding Site Detection (Week 2)
1. **Implement binding site detection**
   - Add PDB parsing with Biopython
   - Implement cavity detection algorithms
   - Add predefined sites for DNA and GSTP1

2. **API endpoints**
   - `/detect-binding-sites` endpoint
   - `/get-predefined-sites` endpoint
   - Integration with molecule processing pipeline

### Phase 3: Docking Integration (Week 3-4)
1. **QAOA-based docking**
   - Implement quantum pose optimization
   - Integration with existing QAOA module
   - Energy function development

2. **Classical docking fallback**
   - AutoDock Vina integration
   - File format conversions (MOL ↔ PDBQT)
   - Results parsing and scoring

3. **Advanced visualization**
   - Binding pose display
   - Interaction highlighting
   - Multiple pose comparison

### Phase 4: Production Integration (Week 5)
1. **Performance optimization**
   - Caching for common targets
   - Background job processing
   - Result storage

2. **User experience**
   - Loading states and progress indicators
   - Error handling and fallbacks
   - Export functionality for results

3. **Documentation and testing**
   - API documentation
   - Unit tests for all components
   - Integration tests with full pipeline

---

## Business Value

### Scientific Impact
- **Visual Validation**: Scientists can see how AI-generated analogs actually bind
- **Mechanism Insights**: Understanding why certain analogs work better
- **Hypothesis Generation**: Visual patterns suggest new optimization directions

### User Experience  
- **Trust Building**: Seeing is believing - visual proof builds confidence
- **Educational**: Helps users understand structure-activity relationships
- **Publication Ready**: High-quality visualizations for papers and presentations

### Competitive Advantage
- **Unique Feature**: Few drug discovery platforms offer integrated quantum + docking
- **End-to-End**: Complete pipeline from generation to binding visualization
- **Customizable**: Support for user-provided targets and custom docking parameters

