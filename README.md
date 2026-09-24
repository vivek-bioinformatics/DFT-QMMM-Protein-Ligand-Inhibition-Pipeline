# Q-MECH
### Quantum Mechanistic Inhibition Pipeline

**Q-MECH** is an open-source, GPU-accelerated Python pipeline for comprehensive quantum chemical and hybrid QM/MM analysis of molecular mechanism of inhibition in protein–ligand complexes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PySCF](https://img.shields.io/badge/PySCF-2.14.0-green.svg)](https://pyscf.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

Q-MECH takes a single protein–ligand complex PDB file as input and automatically performs:

| Step | Analysis | Method |
|------|----------|--------|
| 1 | Binding pocket extraction | Biopython + distance-based (8 Å) |
| 2 | Ligand preparation | RDKit + PySCF gto.Mole |
| 3 | DFT electronic structure | B3LYP/def2-SVP (PySCF + GPU4PySCF) |
| 4 | Frontier molecular orbitals | Real HOMO/LUMO from SCF |
| 5 | Global reactivity descriptors | IP, EA, χ, η, S, ω via Koopmans |
| 6 | Fukui reactivity indices | N/N+1/N−1 DFT states |
| 7 | Molecular electrostatic potential | From DFT electron density |
| 8 | QM/MM interaction energy | PySCF electrostatic embedding |
| 9 | Per-residue decomposition | Electrostatic + Lennard-Jones |
| 10 | H-bond analysis | Geometric + Mulliken charge criterion |
| 11 | Mechanism interpretation | Rule-based from quantum descriptors |
| 12 | Publication figures | 8 × 600 DPI PNG per compound |

---

## Features

- ✅ **Single input** — only a PDB file needed (no SMILES, no topology)
- ✅ **Auto ligand detection** — automatically identifies ligand from HETATM records
- ✅ **GPU acceleration** — via GPU4PySCF (10× faster than CPU)
- ✅ **Real DFT** — actual B3LYP/def2-SVP SCF, not empirical surrogates
- ✅ **Real Fukui** — from N, N+1, N−1 DFT calculations
- ✅ **Real MEP** — from electron density, not point charges
- ✅ **Real QM/MM** — PySCF electrostatic embedding (pyscf.qmmm.mm_charge)
- ✅ **AMBER ff14SB** — protein MM charges from embedded lookup table
- ✅ **Batch mode** — process multiple complexes automatically
- ✅ **Publication figures** — 8 figures per compound at 600 DPI
- ✅ **Structured output** — JSON + TXT reports for downstream analysis

---

## Installation

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (optional but recommended)

### Install dependencies

```bash
# Create conda environment (recommended)
conda create -n qmech python=3.10 -y
conda activate qmech

# Install all dependencies
pip install pyscf biopython rdkit "numpy==1.26.4" scipy matplotlib pandas morfeus-ml

# GPU acceleration (optional — highly recommended)
# For CUDA 12.x:
pip install gpu4pyscf-cuda12x

# For CUDA 11.x:
pip install gpu4pyscf-cuda11x
```

### Verify installation

```bash
python3 - << 'EOF'
import pyscf; print("PySCF:", pyscf.__version__)
from Bio import PDB; print("Biopython: OK")
from rdkit import Chem; print("RDKit: OK")
try:
    from gpu4pyscf import dft; print("GPU4PySCF: OK — GPU mode")
except ImportError:
    from pyscf import dft; print("GPU4PySCF: Not found — CPU mode")
print("Q-MECH ready!")
EOF
```

---

## Usage

### Single complex

```bash
python3 qmech_v1.0.py protein_ligand.pdb LIG results_dir
```

Arguments:
- `protein_ligand.pdb` — PDB file with protein + ligand
- `LIG` — ligand residue name in PDB (use `AUTO` for auto-detection)
- `results_dir` — output directory name

### Or edit the script directly

Open `qmech_v1.0.py` and set at the bottom:

```python
PDB_FILE       = "your_complex.pdb"   # PDB file path
LIGAND_RESNAME = "AUTO"               # or "LIG", "UNK", "ATP" etc.
POCKET_CUTOFF  = 8.0                  # Å — binding pocket radius
N_THREADS      = 24                   # CPU threads
MEP_GRID       = 30                   # MEP grid points per dimension
```

```bash
python3 qmech_v1.0.py
```

### Batch mode (multiple complexes)

Edit `batch_run_qmech.sh`:

```bash
PDBS=(
    "complex1.pdb"
    "complex2.pdb"
    "complex3.pdb"
    "complex4.pdb"
)
LIGAND="AUTO"
```

Run:

```bash
bash batch_run_qmech.sh 2>&1 | tee batch.log
```

### Generate figures

```bash
python3 qmech_figures.py
```

---

## Output

For each compound, Q-MECH generates:

```
results_dir/
├── fig1_homo_lumo.png          ← HOMO/LUMO orbital energy diagram
├── fig2_dft_radar.png          ← DFT reactivity radar (5 descriptors)
├── fig3_fukui.png              ← Fukui f⁺/f⁻/Δf with annotations
├── fig4_mep.png                ← MEP surface features
├── fig5_per_residue.png        ← Per-residue QM/MM energy
├── fig6_elec_vs_lj.png        ← Electrostatic vs vdW decomposition
├── fig7_hbonds.png             ← H-bond distances (color-coded)
├── fig8_mechanism.png          ← Mechanism summary table
├── report.json                 ← Machine-readable results
└── report.txt                  ← Human-readable report
```

For batch runs, a comparison figure is also generated:
```
comparison_all_complexes.png    ← 6-panel comparative figure
```

---

## Computed Parameters

### DFT Global Reactivity Descriptors

| Parameter | Formula | Physical Meaning |
|-----------|---------|-----------------|
| HOMO (eV) | E_HOMO | Electron donation ability |
| LUMO (eV) | E_LUMO | Electron acceptance ability |
| Gap (eV) | E_LUMO − E_HOMO | Kinetic stability / reactivity |
| IP (eV) | −E_HOMO | Ionization potential (Koopmans) |
| EA (eV) | −E_LUMO | Electron affinity (Koopmans) |
| χ (eV) | (IP + EA) / 2 | Electronegativity |
| η (eV) | (IP − EA) / 2 | Chemical hardness |
| S (eV⁻¹) | 1 / (2η) | Global softness |
| ω (eV) | χ² / (2η) | Electrophilicity index |
| ΔNmax | χ / (2η) | Maximum charge transfer |

### Fukui Indices (from N/N+1/N−1 DFT)

| Index | Meaning |
|-------|---------|
| f⁺ | Electrophilic attack sites |
| f⁻ | Nucleophilic attack sites |
| f⁰ | Radical attack sites |
| Δf | Dual descriptor |

---

## Performance

Tested on NVIDIA GeForce RTX 3070 Ti (8 GB VRAM, CUDA 11.8):

| Ligand Size | DFT SCF (GPU) | Fukui (GPU) | QM/MM (CPU) | Total |
|------------|--------------|------------|------------|-------|
| < 30 atoms | ~3 min | ~8 min | ~30 min | ~45 min |
| 30–50 atoms | ~5 min | ~15 min | ~45 min | ~65 min |
| > 50 atoms | ~8 min | ~25 min | ~60 min | ~95 min |

CPU-only (24 threads, 128 GB RAM): approximately 3–4× slower.

---

## Citation

If you use Q-MECH in your research, please cite:

```bibtex
@software{dwivedi_qmech_2026,
  author    = {Dwivedi, Vivek Dhar},
  title     = {{Q-MECH: A GPU-Accelerated Open-Source Python Pipeline 
               for Quantum Mechanical (DFT \& QM/MM) Analysis of 
               Molecular Mechanism of Inhibition in Protein--Ligand 
               Complexes}},
  year      = {2026},
  version   = {1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

### Also cite the underlying engines:

- **PySCF**: Sun et al., *WIREs Comput Mol Sci*, 2018, 8, e1340
- **GPU4PySCF**: Wu et al., *arXiv:2407.09600*, 2024
- **B3LYP**: Becke, *J Chem Phys*, 1993, 98, 5648
- **def2-SVP**: Weigend & Ahlrichs, *PCCP*, 2005, 7, 3297
- **QM/MM**: Warshel & Levitt, *J Mol Biol*, 1976, 103, 227
- **AMBER ff14SB**: Maier et al., *JCTC*, 2015, 11, 3696
- **RDKit**: https://www.rdkit.org

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PySCF | ≥ 2.8.0 | DFT/QM calculations |
| GPU4PySCF | ≥ 1.5.2 | GPU acceleration (optional) |
| RDKit | ≥ 2024.0 | Cheminformatics |
| Biopython | ≥ 1.84 | PDB parsing |
| NumPy | 1.26.4 | Numerical computing |
| SciPy | ≥ 1.10 | Distance calculations |
| Matplotlib | ≥ 3.7 | Figure generation |
| Pandas | ≥ 2.0 | Data management |
| Morfeus-ML | ≥ 0.8 | Steric descriptors |

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Author

**Dr. Vivek Dhar Dwivedi**  
Raja Shankar Shah University, Chindwara, Madhya Pradesh, India  
Computational Drug Design | Molecular Modeling | Bioinformatics

---

## Acknowledgements

The authors acknowledge the developers of PySCF, GPU4PySCF, RDKit, and
Biopython for providing the open-source tools that make Q-MECH possible.
