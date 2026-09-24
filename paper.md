---
title: 'Q-MECH: A GPU-Accelerated Open-Source Python Pipeline for Quantum Mechanical (DFT & QM/MM) Analysis of Molecular Mechanism of Inhibition in Protein–Ligand Complexes'

tags:
  - Python
  - computational chemistry
  - drug discovery
  - DFT
  - QM/MM
  - HOMO-LUMO
  - Fukui indices
  - molecular electrostatic potential
  - protein-ligand interactions
  - GPU acceleration

authors:
  - name: Vivek Dhar Dwivedi
    orcid: 0000-0003-0767-1328
    affiliation: "1, 2"

affiliations:
  - name: Raja Shankar Shah University, Chindwara, Madhya Pradesh, India
    index: 1
  - name: Bioinformatics Research Division, Quanta Calculus, India
    index: 2

date: 24 September 2026

bibliography: paper.bib

---

# Statement of Need

Understanding the molecular mechanism by which small molecules inhibit
protein targets is central to rational drug design. While molecular
docking and molecular dynamics (MD) simulations provide structural
insights into binding modes, they lack the quantum mechanical description
necessary to characterize covalent interactions, electronic reactivity,
and electrostatic complementarity at the electronic structure level
[@sherrill2013; @ryde2016].

Density functional theory (DFT) methods offer rigorous access to frontier
molecular orbital energies, global chemical reactivity descriptors
(ionization potential, electron affinity, electronegativity, chemical
hardness, softness, and electrophilicity index), Fukui reactivity indices,
and molecular electrostatic potential (MEP) that govern protein–ligand
recognition [@yang1986; @parr1995]. Hybrid quantum mechanics/molecular
mechanics (QM/MM) approaches further enable calculation of interaction
energies with explicit inclusion of the protein electrostatic environment
[@warshel1976].

However, the practical application of these methods to drug discovery
workflows has been hindered by: (i) steep learning curves and lack of
automation, (ii) high computational cost, (iii) dependence on expensive
commercial software (e.g., Gaussian, ORCA), and (iv) absence of
integrated, user-friendly pipelines that bridge quantum chemistry with
structural bioinformatics. Existing tools either require SMILES input
(losing binding conformation information), lack QM/MM capability, or do
not support GPU acceleration on consumer hardware.

**Q-MECH** addresses these gaps by providing a fully automated, open-source
Python pipeline that takes a single protein–ligand complex PDB file as the
only required input and produces a comprehensive quantum chemical
characterization of the inhibition mechanism, including publication-ready
figures and structured data reports. GPU acceleration via GPU4PySCF
[@wu2024] reduces computation time from hours to approximately 50 minutes
per complex on a single consumer-grade GPU, making production-grade DFT
calculations accessible to the broader drug discovery community.

# Summary

Q-MECH is a Python pipeline that integrates twelve analysis steps into a
single automated workflow:

1. **Binding pocket extraction** from PDB using Biopython [@cock2009],
   with automatic ligand detection and AMBER ff14SB [@maier2015] charge
   assignment to pocket residues (default cutoff: 8.0 Å).

2. **Ligand preparation** directly from PDB coordinates — no SMILES
   required — with molecular connectivity inferred from van der Waals
   radii and bond orders determined by RDKit [@landrum2013; @bento2020].

3. **Real DFT calculations** at the B3LYP/def2-SVP level [@lee1988;
   @becke1993; @weigend2005] using PySCF [@sun2018; @sun2020] with
   optional GPU acceleration via GPU4PySCF [@wu2024]. Basis set is
   automatically selected based on ligand heavy atom count (def2-SVP for
   ≤60 atoms, STO-3G fallback for convergence failures).

4. **Real frontier molecular orbital energies** (HOMO, LUMO) from the
   converged Kohn–Sham wavefunction, with global chemical reactivity
   descriptors via Koopmans' theorem [@koopmans1934]: ionization potential
   (IP = −E~HOMO~), electron affinity (EA = −E~LUMO~), electronegativity
   (χ), chemical hardness (η), global softness (S), and electrophilicity
   index (ω).

5. **Real Fukui reactivity indices** from three separate DFT calculations
   on the neutral (N), anionic (N+1), and cationic (N−1) electronic states
   [@yang1986], condensed to atoms via Mulliken population analysis: f⁺
   (electrophilic attack sites), f⁻ (nucleophilic attack sites), f⁰
   (radical attack sites), and the dual descriptor Δf = f⁺ − f⁻.

6. **Real molecular electrostatic potential** computed from the DFT
   electron density on a three-dimensional grid surrounding the van der
   Waals surface, including both nuclear and electronic contributions.

7. **QM/MM electrostatic embedding** via PySCF's qmmm module
   (pyscf.qmmm.mm_charge [@sun2018]), modifying the QM Hamiltonian with
   MM point charges: Ĥ~QM/MM~ = Ĥ~QM~ + Σ~M~ Q~M~/|r̂ − R~M~|. Total
   interaction energy includes the embedded QM energy, classical Coulomb
   interaction, and Lennard-Jones van der Waals term.

8. **Per-residue energy decomposition** partitioning electrostatic and van
   der Waals contributions per binding pocket residue, enabling
   identification of pharmacophoric binding interactions.

9. **Protein–ligand hydrogen bond analysis** using geometric criterion
   (donor–acceptor distance ≤ 3.5 Å) with Mulliken charge-based
   donor/acceptor assignment, classified as strong (<3.0 Å), moderate
   (3.0–3.2 Å), or weak (3.2–3.5 Å) [@jeffrey1997].

10. **Rule-based mechanism interpretation** classifying inhibition mode
    (covalent/irreversible, competitive, allosteric), reactivity class,
    electrophilicity, metal chelation potential, and selectivity from
    computed quantum descriptors.

11. **Drug-likeness assessment** using Lipinski's Rule of Five
    [@lipinski2001], Quantitative Estimate of Drug-likeness (QED)
    [@bickerton2012], and topological polar surface area (TPSA), computed
    via RDKit [@rdkit].

12. **Automated publication-quality figure generation** — eight 600 DPI
    PNG figures per compound plus a multi-compound comparative figure —
    using Matplotlib [@hunter2007].

# Implementation

Q-MECH is implemented in Python 3.10+ and organized as a single-file
pipeline (`qmech_v1.0.py`) with a companion batch runner
(`batch_run_qmech.sh`) and figure generator (`qmech_figures.py`). All
dependencies are freely available via pip.

The pipeline supports both GPU (via GPU4PySCF, CUDA 11.x/12.x) and CPU
execution, with automatic fallback to CPU mode when GPU is unavailable.
Multiple complexes can be processed sequentially in batch mode, with each
complex producing an independent result directory containing JSON and
plain-text reports alongside all figures.

## Performance

On an NVIDIA GeForce RTX 3070 Ti (8 GB VRAM, CUDA 11.8) with an Intel
24-thread CPU and 128 GB RAM, Q-MECH processes a typical drug-like
compound (30–50 heavy atoms) in approximately 65 minutes: DFT SCF (~5
min GPU), Fukui N/N+1/N−1 (~15 min GPU), QM/MM (~45 min CPU), and
figure generation (~2 min). CPU-only execution requires approximately 3–4×
longer. Note that the QM/MM step runs on CPU as GPU4PySCF does not
currently support the `pyscf.qmmm.mm_charge` interface.

## Output

For each compound, Q-MECH produces the following files in the output
directory:

| File | Content |
|------|---------|
| `fig1_homo_lumo.png` | HOMO/LUMO orbital energy diagram |
| `fig2_dft_radar.png` | DFT reactivity radar chart (5 descriptors) |
| `fig3_fukui.png` | Fukui f⁺/f⁻/Δf indices with reactive site annotations |
| `fig4_mep.png` | MEP surface feature bar chart |
| `fig5_per_residue.png` | Per-residue QM/MM energy decomposition |
| `fig6_elec_vs_lj.png` | Electrostatic vs van der Waals contributions |
| `fig7_hbonds.png` | Protein–ligand hydrogen bond distance chart |
| `fig8_mechanism.png` | Inhibition mechanism summary table |
| `report.json` | Machine-readable structured results |
| `report.txt` | Human-readable text report |

# AI Disclosure

Portions of the Q-MECH source code and manuscript text were drafted with
the assistance of Claude (Anthropic, claude.ai), an AI language model.
All AI-generated content was critically reviewed, validated, and edited
by the author. The scientific design, conceptualization, validation, and
interpretation of results are entirely the work of the author. The
computational results reported herein were generated by running the
pipeline on real protein–ligand complex data, and all quantum chemical
calculations were performed using established open-source software
(PySCF, GPU4PySCF) without AI involvement in the actual calculations.

# Acknowledgements

The author acknowledges the developers of PySCF [@sun2018; @sun2020],
GPU4PySCF [@wu2024], RDKit [@rdkit], and Biopython [@cock2009] for
providing the open-source computational chemistry and cheminformatics
tools that form the foundation of Q-MECH.

# References
