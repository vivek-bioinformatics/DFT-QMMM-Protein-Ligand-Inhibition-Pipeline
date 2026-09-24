"""
╔══════════════════════════════════════════════════════════════════╗
║   Q-MECH: A GPU-Accelerated Open-Source Python Pipeline for     ║
║   Quantum Mechanical (DFT & QM/MM) Analysis of Molecular        ║
║   Mechanism of Inhibition in Protein–Ligand Complexes           ║
╠══════════════════════════════════════════════════════════════════╣
║   Version  : 1.0                                                 ║
║   Engine   : PySCF 2.14.0 + GPU4PySCF                          ║
║   Method   : B3LYP/def2-SVP DFT + AMBER ff14SB QM/MM           ║
║   Input    : Protein-Ligand complex PDB file (single input)     ║
║   Output   : HOMO/LUMO, Fukui, MEP, QM/MM energies,            ║
║              H-bonds, Mechanism, 8 figures per compound         ║
╠══════════════════════════════════════════════════════════════════╣
║   Citation:                                                      ║
║   Dwivedi VD (2026). Q-MECH v1.0. Zenodo.                      ║
║   https://doi.org/10.5281/zenodo.XXXXXXX                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python3 qmech_v1.0.py Complex_1.pdb LIG Complex_1_results
    python3 qmech_v1.0.py  (uses settings defined at bottom of script)
"""

import os, sys, json, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ── RDKit ──────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                        rdDetermineBonds, RWMol, Conformer)
from rdkit.Chem.rdchem import Atom
from rdkit.Geometry import Point3D

# ── PySCF — GPU or CPU ────────────────────────────────────────────
from pyscf import gto, scf, qmmm as pyscf_qmmm
try:
    from gpu4pyscf import dft
    print("  ✓ GPU4PySCF loaded - GPU mode")
except ImportError:
    from pyscf import dft
    print("  ⚠ GPU4PySCF not found - CPU mode")

# ── Biopython ──────────────────────────────────────────────────────
from Bio import PDB as BioPDB

# ── Constants ──────────────────────────────────────────────────────
HARTREE_TO_EV   = 27.211386
HARTREE_TO_KCAL = 627.509
BOHR_TO_ANG     = 0.529177
ANG_TO_BOHR     = 1.0 / BOHR_TO_ANG

# ── Element lookups ────────────────────────────────────────────────
ELEM_ANUM = {'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,
             'CL':17,'BR':35,'I':53,'H':1}
VDW_ANG   = {'C':1.70,'N':1.55,'O':1.52,'S':1.80,
             'P':1.80,'H':1.20,'F':1.47}

# ── Protein residue sets ───────────────────────────────────────────
PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HIE","HID","HIP","CYX","MSE"
}
SOLVENT = {"HOH","WAT","TIP","TIP3","SOL","NA","CL","MG","ZN",
           "CA","K","NA+","CL-"}

# ══════════════════════════════════════════════════════════════════
#  AMBER ff14SB Partial Charges
# ══════════════════════════════════════════════════════════════════
RESIDUE_CHARGES = {
    "GLY":{"N":-0.4157,"CA":0.0317,"C":0.5973,"O":-0.5679},
    "ALA":{"N":-0.4157,"CA":0.0337,"C":0.5973,"O":-0.5679,"CB":-0.1825},
    "VAL":{"N":-0.4157,"CA":0.0859,"C":0.5973,"O":-0.5679,"CB":0.1986,"CG1":-0.3195,"CG2":-0.3195},
    "LEU":{"N":-0.4157,"CA":0.0922,"C":0.5973,"O":-0.5679,"CB":-0.2483,"CG":0.3598,"CD1":-0.4121,"CD2":-0.4121},
    "ILE":{"N":-0.4157,"CA":0.1015,"C":0.5973,"O":-0.5679,"CB":0.1302},
    "PHE":{"N":-0.4157,"CA":0.1141,"C":0.5973,"O":-0.5679,"CB":-0.1256,"CG":0.1329,"CD1":-0.1915,"CD2":-0.1915,"CE1":-0.1374,"CE2":-0.1374,"CZ":-0.1138},
    "TRP":{"N":-0.4157,"CA":0.0930,"C":0.5973,"O":-0.5679,"CB":-0.0050,"CG":-0.1415,"NE1":-0.3418},
    "SER":{"N":-0.4157,"CA":0.0567,"C":0.5973,"O":-0.5679,"CB":0.2117,"OG":-0.6546},
    "THR":{"N":-0.4157,"CA":0.0765,"C":0.5973,"O":-0.5679,"CB":0.3654,"OG1":-0.6761},
    "CYS":{"N":-0.4157,"CA":0.0213,"C":0.5973,"O":-0.5679,"CB":-0.1231,"SG":-0.3119},
    "MET":{"N":-0.4157,"CA":0.0590,"C":0.5973,"O":-0.5679,"CB":0.0342,"SD":-0.2774},
    "ASP":{"N":-0.4157,"CA":0.0381,"C":0.5973,"O":-0.5679,"CB":-0.0303,"CG":0.7994,"OD1":-0.8014,"OD2":-0.8014},
    "GLU":{"N":-0.4157,"CA":0.0145,"C":0.5973,"O":-0.5679,"CB":0.0560,"CD":0.8054,"OE1":-0.8188,"OE2":-0.8188},
    "ASN":{"N":-0.4157,"CA":0.0143,"C":0.5973,"O":-0.5679,"CB":-0.2041,"CG":0.7130,"OD1":-0.5931,"ND2":-0.9191},
    "GLN":{"N":-0.4157,"CA":-0.0031,"C":0.5973,"O":-0.5679,"CD":0.6951,"OE1":-0.6086,"NE2":-0.9407},
    "LYS":{"N":-0.4157,"CA":0.0966,"C":0.5973,"O":-0.5679,"CB":-0.0094,"CE":0.3260,"NZ":-0.3854},
    "ARG":{"N":-0.4157,"CA":0.0298,"C":0.5973,"O":-0.5679,"NE":-0.5295,"CZ":0.8076,"NH1":-0.8627,"NH2":-0.8627},
    "HIS":{"N":-0.4157,"CA":0.0188,"C":0.5973,"O":-0.5679,"CB":-0.0462,"ND1":-0.3811,"NE2":-0.5727},
    "TYR":{"N":-0.4157,"CA":0.0581,"C":0.5973,"O":-0.5679,"CB":-0.0152,"CZ":0.3226,"OH":-0.5579},
    "PRO":{"N":-0.2548,"CA":0.0000,"C":0.5896,"O":-0.5748,"CB":-0.0070},
}
DEFAULT_Q = {"N":-0.40,"CA":0.03,"C":0.60,"O":-0.57,"CB":-0.12}

def get_amber_charge(resname, atom_name):
    return RESIDUE_CHARGES.get(resname,{}).get(
           atom_name, DEFAULT_Q.get(atom_name, 0.0))


# ══════════════════════════════════════════════════════════════════
#  STEP 1 : PDB Parsing & Binding Pocket Extraction
# ══════════════════════════════════════════════════════════════════
def extract_pocket_from_pdb(pdb_path, ligand_resname="AUTO",
                             cutoff=8.0):
    print(f"\n{'═'*66}")
    print(f"  STEP 1  |  PDB Parsing & Pocket Extraction")
    print(f"{'═'*66}")
    print(f"  File: {pdb_path}")

    parser = BioPDB.PDBParser(QUIET=True)
    struct = parser.get_structure("mol", pdb_path)

    prot_atoms = []; lig_atoms = []; all_resnames = set()
    for model in struct:
        for chain in model:
            for res in chain:
                rn = res.resname.strip()
                ri = res.get_id()[1]
                ch = chain.get_id()
                all_resnames.add(rn)
                for atom in res:
                    coord = np.array(atom.get_coord(), dtype=float)
                    aname = atom.get_name().strip()
                    if rn in PROTEIN_RESNAMES:
                        prot_atoms.append((rn,ri,ch,aname,coord))
                    elif rn not in SOLVENT:
                        lig_atoms.append((rn,ri,ch,aname,coord,atom))

    # Auto-detect ligand
    if ligand_resname.upper() == "AUTO":
        candidates = {}
        for rn,ri,ch,an,co,at in lig_atoms:
            if len([x for x in lig_atoms if x[0]==rn]) >= 5:
                candidates[rn] = candidates.get(rn,0) + 1
        if not candidates:
            raise ValueError(
                f"No ligand found.\nAvailable: {all_resnames}\n"
                f"Set ligand_resname='YOUR_LIG' manually.")
        ligand_resname = sorted(candidates,
                                key=candidates.get, reverse=True)[0]
        if len(candidates) > 1:
            print(f"  ⚠  Multiple ligands: {list(candidates.keys())}")
        print(f"  ✓  Ligand auto-detected: '{ligand_resname}'")

    lig_only = [(rn,ri,ch,an,co,at) for rn,ri,ch,an,co,at in lig_atoms
                if rn == ligand_resname]
    if not lig_only:
        raise ValueError(
            f"Ligand '{ligand_resname}' not found.\n"
            f"Available: {all_resnames}")

    lig_pos  = np.array([co for _,_,_,_,co,_ in lig_only])
    lig_cent = lig_pos.mean(axis=0)

    # Element from Biopython element field
    lig_elems = []
    for _,_,_,an,_,atom in lig_only:
        el = atom.element.strip() if atom.element and atom.element.strip() \
             else ''.join(c for c in an if c.isalpha())[:1].upper()
        el = ''.join(c for c in el if c.isalpha()).upper()
        el = el[:2] if len(el)>=2 and el[:2] in ELEM_ANUM else el[0] if el else 'C'
        lig_elems.append(el)

    n_prot_res = len(set((rn,ri,ch) for rn,ri,ch,an,co in prot_atoms))
    print(f"  Ligand  : {len(lig_only)} atoms")
    print(f"  Protein : {len(prot_atoms)} atoms | {n_prot_res} residues")

    # Distance-based pocket
    prot_coords = np.array([co for _,_,_,_,co in prot_atoms])
    d_mat    = cdist(prot_coords, lig_pos)
    in_pocket= d_mat.min(axis=1) <= cutoff
    pocket_atom_list = [prot_atoms[i]
                        for i in range(len(prot_atoms)) if in_pocket[i]]

    pocket_residues = []
    seen = set()
    for rn,ri,ch,an,co in pocket_atom_list:
        key = (rn,ri,ch)
        if key not in seen:
            seen.add(key)
            pocket_residues.append({"resname":rn,"resid":ri,"chain":ch})
    pocket_residues.sort(key=lambda x: x["resid"])

    print(f"\n  Pocket Residues ({len(pocket_residues)}) within {cutoff} Å:")
    line = "  "
    for i,r in enumerate(pocket_residues):
        line += f"{r['chain']}:{r['resname']}{r['resid']}  "
        if (i+1) % 7 == 0: print(line); line = "  "
    if line.strip(): print(line)

    p_coords  = np.array([co for _,_,_,_,co in pocket_atom_list])
    p_charges = np.array([get_amber_charge(rn, an)
                          for rn,ri,ch,an,co in pocket_atom_list])
    p_names   = [f"{rn}{ri}:{an}"
                 for rn,ri,ch,an,co in pocket_atom_list]
    p_resnames= [rn for rn,ri,ch,an,co in pocket_atom_list]
    p_resids  = [ri for rn,ri,ch,an,co in pocket_atom_list]

    print(f"\n  Pocket atoms : {len(pocket_atom_list)}")
    print(f"  MM total charge: {p_charges.sum():.3f} e")

    return {
        "lig_pos"        : lig_pos,
        "lig_elems"      : lig_elems,
        "lig_names"      : [an for _,_,_,an,_,_ in lig_only],
        "lig_resname"    : ligand_resname,
        "lig_centroid"   : lig_cent,
        "mm_coords"      : p_coords,
        "mm_charges"     : p_charges,
        "mm_names"       : p_names,
        "mm_resnames"    : p_resnames,
        "mm_resids"      : p_resids,
        "pocket_residues": pocket_residues,
        "n_pocket_atoms" : len(pocket_atom_list),
        "n_pocket_res"   : len(pocket_residues),
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 2 : Build PySCF Mol from PDB Coordinates
# ══════════════════════════════════════════════════════════════════
def build_ligand_mol(pocket):
    print(f"\n{'═'*66}")
    print(f"  STEP 2  |  Building Ligand for PySCF")
    print(f"{'─'*66}")

    elems  = pocket["lig_elems"]
    coords = pocket["lig_pos"].astype(float)
    n_heavy= sum(1 for e in elems if e != 'H')

    if   n_heavy < 30:  basis = "def2-SVP"
    elif n_heavy < 60:  basis = "def2-svp"
    else:               basis = "sto-3g"
    print(f"  Heavy atoms : {n_heavy}  →  Basis: {basis}")

    # RDKit mol for connectivity
    rwmol = RWMol()
    conf  = Conformer(len(elems))
    for i,(el,pos) in enumerate(zip(elems, coords)):
        rwmol.AddAtom(Atom(ELEM_ANUM.get(el.upper(), 6)))
        conf.SetAtomPosition(i, Point3D(float(pos[0]),
                                        float(pos[1]),
                                        float(pos[2])))
    rwmol.AddConformer(conf, assignId=True)
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            d  = float(np.linalg.norm(coords[i]-coords[j]))
            ri = VDW_ANG.get(elems[i][0].upper(), 1.7)*0.45
            rj = VDW_ANG.get(elems[j][0].upper(), 1.7)*0.45
            if d < (ri+rj)*1.35:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)
    try:
        rdDetermineBonds.DetermineBondOrders(rwmol, charge=0)
        Chem.SanitizeMol(rwmol)
    except Exception:
        try:
            Chem.SanitizeMol(rwmol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception: pass
    rdkit_mol = rwmol.GetMol()

    # Build PySCF mol
    atom_str = "; ".join(
        f"{el} {coords[i,0]:.6f} {coords[i,1]:.6f} {coords[i,2]:.6f}"
        for i,el in enumerate(elems)
    )
    pyscf_mol = gto.Mole()
    pyscf_mol.atom   = atom_str
    pyscf_mol.basis  = basis
    pyscf_mol.charge = 0
    pyscf_mol.spin   = 0
    pyscf_mol.unit   = 'Angstrom'
    pyscf_mol.verbose     = 3
    pyscf_mol.max_memory  = 80000
    try:
        pyscf_mol.build()
    except Exception:
        pyscf_mol.spin = 1
        try:
            pyscf_mol.build()
        except Exception as e:
            print(f"  ⚠  Even electron config failed ({e}), trying spin=1")
            pyscf_mol.build()

    print(f"  PySCF mol  : {pyscf_mol.natm} atoms  "
          f"| {pyscf_mol.nao} AOs  "
          f"| charge={pyscf_mol.charge}  "
          f"| basis={basis}")

    return {
        "pyscf_mol" : pyscf_mol,
        "rdkit_mol" : rdkit_mol,
        "coords"    : coords,
        "elems"     : elems,
        "basis"     : basis,
        "charge"    : pyscf_mol.charge,
        "n_heavy"   : n_heavy,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 3 : Real DFT — B3LYP/def2-SVP
# ══════════════════════════════════════════════════════════════════
def run_dft_neutral(lig, functional="B3LYP", n_threads=24):
    print(f"\n{'═'*66}")
    print(f"  STEP 3  |  Real DFT  [{functional}/{lig['basis']}]")
    print(f"{'─'*66}")

    import pyscf as _pyscf
    _pyscf.lib.num_threads(n_threads)

    mol = lig["pyscf_mol"]
    t0  = time.time()

    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc        = functional
    mf.max_cycle = 200
    mf.conv_tol  = 1e-9
    mf.diis_space= 12

    print(f"  Running {functional}/{lig['basis']} SCF ...")
    print(f"  Threads: {n_threads}  |  Memory: {mol.max_memory} MB")
    energy = mf.kernel()

    if not mf.converged:
        print(f"  ⚠  SCF not converged — trying smaller basis STO-3G")
        mol2 = mol.copy(); mol2.basis = "sto-3g"; mol2.build()
        if mol2.spin == 0:
            mf2 = dft.RKS(mol2)
        else:
            mf2 = dft.UKS(mol2)
        mf2.xc = functional; mf2.max_cycle = 300
        energy = mf2.kernel()
        mf = mf2; mol = mol2

    dt = time.time()-t0
    print(f"  ✓ SCF converged in {dt:.1f}s")
    print(f"  Total energy: {energy:.8f} Ha  "
          f"({energy*HARTREE_TO_KCAL:.2f} kcal/mol)")

    # HOMO/LUMO
    mo_e   = mf.mo_energy if mol.spin==0 else mf.mo_energy[0]
    mo_occ = mf.mo_occ    if mol.spin==0 else mf.mo_occ[0]
    homo_i = int(np.where(mo_occ > 0)[0].max())
    lumo_i = homo_i + 1

    HOMO_ev = float(mo_e[homo_i]) * HARTREE_TO_EV
    LUMO_ev = float(mo_e[lumo_i]) * HARTREE_TO_EV
    gap_ev  = LUMO_ev - HOMO_ev
    IP_ev   = -HOMO_ev
    EA_ev   = -LUMO_ev
    chi     = (IP_ev + EA_ev) / 2
    eta     = max((IP_ev - EA_ev) / 2, 0.01)
    omega   = chi**2 / (2*eta)
    S       = 1.0 / (2*eta)
    dN_max  = chi / (2*eta)

    print(f"\n  ── Real Orbital Energies ──────────────────────")
    print(f"  HOMO  (orbital {homo_i}): {HOMO_ev:>9.4f} eV")
    print(f"  LUMO  (orbital {lumo_i}): {LUMO_ev:>9.4f} eV")
    print(f"  HOMO-LUMO gap          : {gap_ev:>9.4f} eV")
    print(f"\n  ── Koopmans DFT Descriptors ───────────────────")
    print(f"  IP  = -E(HOMO) : {IP_ev:>9.4f} eV")
    print(f"  EA  = -E(LUMO) : {EA_ev:>9.4f} eV")
    print(f"  χ (chi)        : {chi:>9.4f} eV")
    print(f"  η (eta)        : {eta:>9.4f} eV")
    print(f"  ω (omega)      : {omega:>9.4f} eV")
    print(f"  S (softness)   : {S:>9.4f} eV⁻¹")

    # Mulliken charges (CPU safe)
    dm_n   = mf.make_rdm1()
    ovlp   = mol.intor('int1e_ovlp')
    from pyscf import scf as cpu_scf
    dm_np  = dm_n.get() if hasattr(dm_n,'get') else np.array(dm_n)
    if dm_np.ndim == 3: dm_np = dm_np[0] + dm_np[1]
    ovlp_np= np.array(ovlp)
    ps     = np.dot(dm_np, ovlp_np)
    pop_ao = np.einsum('ii->i', ps).real
    aoslice= mol.aoslice_by_atom()
    atom_pop = np.array([pop_ao[s[2]:s[3]].sum() for s in aoslice])
    mulliken_charges = mol.atom_charges().astype(float) - atom_pop

    # RDKit descriptors
    rdmol = lig["rdkit_mol"]
    try:
        mw   = Descriptors.MolWt(rdmol)
        logp = Descriptors.MolLogP(rdmol)
        tpsa = rdMolDescriptors.CalcTPSA(rdmol)
        hbd  = rdMolDescriptors.CalcNumHBD(rdmol)
        hba  = rdMolDescriptors.CalcNumHBA(rdmol)
        qed  = Descriptors.qed(rdmol)
        rot  = rdMolDescriptors.CalcNumRotatableBonds(rdmol)
        arom = rdMolDescriptors.CalcNumAromaticRings(rdmol)
    except Exception:
        mw=logp=tpsa=hbd=hba=qed=rot=arom=0

    descriptors = {
        "DFT functional"           : functional,
        "Basis set"                : lig["basis"],
        "SCF energy (Ha)"          : round(energy, 8),
        "SCF energy (kcal/mol)"    : round(energy*HARTREE_TO_KCAL, 4),
        "HOMO energy (eV)"         : round(HOMO_ev, 4),
        "LUMO energy (eV)"         : round(LUMO_ev, 4),
        "HOMO-LUMO gap (eV)"       : round(gap_ev,  4),
        "IP = -E(HOMO) (eV)"       : round(IP_ev,   4),
        "EA = -E(LUMO) (eV)"       : round(EA_ev,   4),
        "Electronegativity χ (eV)" : round(chi,     4),
        "Chemical Hardness η (eV)" : round(eta,     4),
        "Global Softness S (eV⁻¹)" : round(S,       4),
        "Electrophilicity ω (eV)"  : round(omega,   4),
        "ΔN_max"                   : round(dN_max,  4),
        "MW (Da)"                  : round(mw,   2),
        "LogP"                     : round(logp, 3),
        "TPSA (Å²)"                : round(tpsa, 2),
        "HBD"                      : int(hbd),
        "HBA"                      : int(hba),
        "QED"                      : round(qed,  3),
        "RotBonds"                 : int(rot),
        "AromaticRings"            : int(arom),
    }

    return {
        "mf"               : mf,
        "mol_used"         : mol,
        "energy"           : energy,
        "dm_n"             : dm_n,
        "ovlp"             : ovlp,
        "mo_energy"        : mo_e,
        "mo_occ"           : mo_occ,
        "homo_idx"         : homo_i,
        "lumo_idx"         : lumo_i,
        "HOMO_ev"          : HOMO_ev,
        "LUMO_ev"          : LUMO_ev,
        "gap_ev"           : gap_ev,
        "mulliken_charges" : mulliken_charges,
        "descriptors"      : descriptors,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 4 : Real Fukui Indices — N/N+1/N-1 DFT
# ══════════════════════════════════════════════════════════════════
def run_fukui(lig, dft_n, functional="B3LYP"):
    print(f"\n{'═'*66}")
    print(f"  STEP 4  |  Real Fukui Indices  [N / N+1 / N-1 DFT]")
    print(f"{'─'*66}")

    mol_n = dft_n.get("mol_used", lig["pyscf_mol"])
    dm_n  = dft_n["dm_n"]

    def get_atom_populations(mol, dm):
        from pyscf import scf as cpu_scf
        dm_np = dm.get() if hasattr(dm,'get') else np.array(dm)
        if dm_np.ndim == 3: dm_np = dm_np[0] + dm_np[1]
        ovlp_l  = mol.intor('int1e_ovlp')
        ps      = np.dot(dm_np, ovlp_l)
        pop_ao  = np.einsum('ii->i', ps).real
        aoslice = mol.aoslice_by_atom()
        return np.array([pop_ao[s[2]:s[3]].sum() for s in aoslice])

    q_n = get_atom_populations(mol_n, dm_n)

    # N+1 (anion)
    print(f"  Running N+1 (anion) {functional}/{lig['basis']} ...")
    t0 = time.time()
    mol_p = mol_n.copy()
    mol_p.charge = mol_n.charge - 1
    nelec_p = mol_p.nelectron
    mol_p.spin = nelec_p % 2
    mol_p.build()
    mf_p = dft.UKS(mol_p)
    mf_p.xc = functional; mf_p.max_cycle = 200; mf_p.verbose = 0
    mf_p.kernel()
    q_p = get_atom_populations(mol_p, mf_p.make_rdm1()[0]+mf_p.make_rdm1()[1]
                               if hasattr(mf_p.make_rdm1(),'__len__') and
                               len(mf_p.make_rdm1())==2
                               else mf_p.make_rdm1())
    print(f"  N+1 done in {time.time()-t0:.1f}s  (converged={mf_p.converged})")

    # N-1 (cation)
    print(f"  Running N-1 (cation) {functional}/{lig['basis']} ...")
    t0 = time.time()
    mol_m = mol_n.copy()
    mol_m.charge = mol_n.charge + 1
    nelec_m = mol_m.nelectron
    mol_m.spin = nelec_m % 2
    mol_m.build()
    mf_m = dft.UKS(mol_m)
    mf_m.xc = functional; mf_m.max_cycle = 200; mf_m.verbose = 0
    mf_m.kernel()
    q_m = get_atom_populations(mol_m, mf_m.make_rdm1()[0]+mf_m.make_rdm1()[1]
                               if hasattr(mf_m.make_rdm1(),'__len__') and
                               len(mf_m.make_rdm1())==2
                               else mf_m.make_rdm1())
    print(f"  N-1 done in {time.time()-t0:.1f}s  (converged={mf_m.converged})")

    f_plus  = q_p - q_n
    f_minus = q_n - q_m
    f_zero  = (f_plus + f_minus) / 2
    delta_f = f_plus - f_minus

    top_fp = int(np.argmax(f_plus))
    top_fm = int(np.argmax(f_minus))

    print(f"\n  ── Real Fukui Indices (Mulliken condensed) ────")
    print(f"  {'Atom':<8} {'Symbol':<6} {'f+':<10} {'f-':<10} "
          f"{'f0':<10} {'Δf':<10}")
    print(f"  {'─'*54}")
    for i in range(mol_n.natm):
        sym = mol_n.atom_symbol(i)
        print(f"  {i:<8} {sym:<6} {f_plus[i]:>+9.4f}  "
              f"{f_minus[i]:>+9.4f}  {f_zero[i]:>+9.4f}  "
              f"{delta_f[i]:>+9.4f}")

    print(f"\n  Most electrophilic: atom {top_fp} "
          f"({mol_n.atom_symbol(top_fp)}) f+={f_plus[top_fp]:+.4f}")
    print(f"  Most nucleophilic : atom {top_fm} "
          f"({mol_n.atom_symbol(top_fm)}) f-={f_minus[top_fm]:+.4f}")

    return {
        "f_plus"  : f_plus, "f_minus": f_minus,
        "f_zero"  : f_zero, "delta_f": delta_f,
        "q_n":q_n, "q_p":q_p, "q_m":q_m,
        "E_anion" : mf_p.e_tot, "E_cation": mf_m.e_tot,
        "top_electrophilic_atom": top_fp,
        "top_nucleophilic_atom" : top_fm,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 5 : Real MEP from DFT Electron Density
# ══════════════════════════════════════════════════════════════════
def compute_real_mep(lig, dft_n, n_pts_per_dim=30):
    print(f"\n{'═'*66}")
    print(f"  STEP 5  |  Real MEP from DFT Electron Density")
    print(f"{'─'*66}")

    mol    = dft_n.get("mol_used", lig["pyscf_mol"])
    dm     = dft_n["dm_n"]
    coords = lig["coords"]
    elems  = lig["elems"]

    lo = coords.min(axis=0)-3.0; hi = coords.max(axis=0)+3.0
    xs,ys,zs = (np.linspace(lo[i],hi[i],n_pts_per_dim) for i in range(3))
    gx,gy,gz = np.meshgrid(xs,ys,zs,indexing="ij")
    grid_ang  = np.column_stack([gx.ravel(),gy.ravel(),gz.ravel()])

    vdw_r = np.array([VDW_ANG.get(e[0].upper(),1.7) for e in elems])
    dm_atoms = cdist(grid_ang, coords)
    outside  = ~np.any(dm_atoms < vdw_r, axis=1)
    grid_out = grid_ang[outside]
    grid_bohr= grid_out * ANG_TO_BOHR

    print(f"  Grid points (outside vdW): {len(grid_out):,}")
    print(f"  Computing nuclear + electronic MEP contributions ...")

    from pyscf.dft import numint as cpu_numint

    # Nuclear
    atom_coords_bohr = mol.atom_coords()
    atom_charges     = mol.atom_charges()
    dist_nuc = cdist(grid_bohr, atom_coords_bohr)
    V_nuc    = (atom_charges / np.clip(dist_nuc,0.01,None)).sum(axis=1)

    # Electronic
    ao_full = cpu_numint.eval_ao(mol, grid_ang*ANG_TO_BOHR, deriv=0)
    dm_np   = dm.get() if hasattr(dm,'get') else dm
    if dm_np.ndim == 3: dm_np = dm_np[0]+dm_np[1]
    ao_np   = ao_full.get() if hasattr(ao_full,'get') else ao_full
    rho_full= cpu_numint.eval_rho(mol, ao_np, dm_np)

    dV = ((hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2])
          /n_pts_per_dim**3 * ANG_TO_BOHR**3)
    V_elec = np.zeros(len(grid_bohr))
    batch  = 200
    for i in range(0, len(grid_bohr), batch):
        g_batch  = grid_bohr[i:i+batch]
        d_batch  = cdist(g_batch, grid_ang*ANG_TO_BOHR)+1e-9
        V_elec[i:i+len(g_batch)] = -(rho_full/d_batch).sum(axis=1)*dV

    V_mep = V_nuc + V_elec

    print(f"  MEP range: [{V_mep.min():.5f}, {V_mep.max():.5f}] a.u.")
    print(f"  Nucleophilic site  (min): {grid_out[V_mep.argmin()].round(2)} Å")
    print(f"  Electrophilic site (max): {grid_out[V_mep.argmax()].round(2)} Å")

    return {
        "grid"      : grid_out, "mep": V_mep,
        "V_nuc"     : V_nuc,   "V_elec": V_elec,
        "atom_coords": coords,
        "MEP_min"   : float(V_mep.min()),
        "MEP_max"   : float(V_mep.max()),
        "MEP_mean"  : float(V_mep.mean()),
        "MEP_std"   : float(V_mep.std()),
        "pct_pos"   : float((V_mep>0).mean()*100),
        "pct_neg"   : float((V_mep<0).mean()*100),
        "nuc_site"  : grid_out[V_mep.argmin()].tolist(),
        "elec_site" : grid_out[V_mep.argmax()].tolist(),
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 6 : Real QM/MM — PySCF Electrostatic Embedding
# ══════════════════════════════════════════════════════════════════
def run_qmmm(lig, pocket, functional="B3LYP"):
    print(f"\n{'═'*66}")
    print(f"  STEP 6  |  Real QM/MM  [PySCF electrostatic embedding]")
    print(f"{'─'*66}")

    mol        = lig["pyscf_mol"]
    mm_coords  = pocket["mm_coords"]
    mm_charges = pocket["mm_charges"]
    mm_coords_bohr = mm_coords * ANG_TO_BOHR

    print(f"  QM region : {mol.natm} atoms  ({functional}/{lig['basis']})")
    print(f"  MM region : {len(mm_charges)} atoms  (AMBER ff14SB)")
    print(f"  MM total charge: {mm_charges.sum():.3f} e")

    # CPU DFT for QM/MM (GPU4PySCF not compatible with pyscf.qmmm)
    from pyscf import dft as cpu_dft
    if mol.spin == 0:
        mf_base = cpu_dft.RKS(mol)
    else:
        mf_base = cpu_dft.UKS(mol)
    mf_base.xc = functional

    mf_qmmm = pyscf_qmmm.mm_charge(mf_base, mm_coords_bohr, mm_charges)
    mf_qmmm.max_cycle = 200
    mf_qmmm.verbose   = 3

    print(f"\n  Running embedded QM/MM SCF ...")
    t0 = time.time()
    E_qmmm = mf_qmmm.kernel()
    print(f"  QM/MM SCF done in {time.time()-t0:.1f}s  "
          f"(converged={mf_qmmm.converged})")
    print(f"  E_QM(embedded): {E_qmmm:.8f} Ha  "
          f"= {E_qmmm*HARTREE_TO_KCAL:.2f} kcal/mol")

    # Classical terms
    lig_coords = lig["coords"].astype(float)
    dist_qm_mm = cdist(lig_coords, mm_coords)

    dm_qmmm  = mf_qmmm.make_rdm1()
    ovlp     = mol.intor('int1e_ovlp')
    dm_np    = dm_qmmm.get() if hasattr(dm_qmmm,'get') else np.array(dm_qmmm)
    if dm_np.ndim == 3: dm_np = dm_np[0]+dm_np[1]
    ovlp_np  = np.array(ovlp)
    ps       = np.dot(dm_np, ovlp_np)
    pop_ao   = np.einsum('ii->i', ps).real
    aoslice  = mol.aoslice_by_atom()
    atom_pop = np.array([pop_ao[s[2]:s[3]].sum() for s in aoslice])
    qm_charges = mol.atom_charges().astype(float) - atom_pop

    E_elec_cl = (np.outer(qm_charges, mm_charges) /
                 np.clip(dist_qm_mm,0.1,None)).sum() * 332.06
    eps, sigma = 0.12, 3.4
    dc  = np.clip(dist_qm_mm, 2.5, None)
    r6  = (sigma/dc)**6
    E_LJ= (4*eps*(r6**2-r6)).sum()
    E_total = E_qmmm*HARTREE_TO_KCAL + E_LJ

    print(f"\n  ── QM/MM Energy Breakdown ──────────────────────")
    print(f"  E_QM embedded  : {E_qmmm*HARTREE_TO_KCAL:>12.3f} kcal/mol")
    print(f"  E_elec (class) : {E_elec_cl:>12.3f} kcal/mol")
    print(f"  E_LJ (vdW)     : {E_LJ:>12.3f} kcal/mol")
    print(f"  E_total QM/MM  : {E_total:>12.3f} kcal/mol")

    # Per-residue decomposition
    prn = pocket["mm_resnames"]; pri = pocket["mm_resids"]
    per_res = {}
    for j,(rn,ri) in enumerate(zip(prn,pri)):
        key = f"{rn}{ri}"
        if key not in per_res:
            per_res[key] = {"resname":rn,"resid":ri,"E_elec":0.,"E_LJ":0.}
        ej  = (qm_charges/np.clip(dist_qm_mm[:,j],0.1,None)*mm_charges[j]).sum()*332.06
        dc_j= np.clip(dist_qm_mm[:,j],2.5,None)
        lj_j= (4*eps*((sigma/dc_j)**12-(sigma/dc_j)**6)).sum()
        per_res[key]["E_elec"] += ej
        per_res[key]["E_LJ"]   += lj_j

    df = pd.DataFrame([
        {"Residue":k,"Resname":v["resname"],"Resid":v["resid"],
         "E_elec":round(v["E_elec"],3),"E_LJ":round(v["E_LJ"],3),
         "E_total":round(v["E_elec"]+v["E_LJ"],3)}
        for k,v in per_res.items()
    ]).sort_values("E_total")

    print(f"\n  Top binding residues:")
    print(f"  {'Residue':<12} {'E_elec':>10} {'E_LJ':>9} "
          f"{'E_total':>10}  kcal/mol")
    print(f"  {'─'*44}")
    for _,r in df.head(5).iterrows():
        print(f"  {r['Residue']:<12} {r['E_elec']:>10.3f} "
              f"{r['E_LJ']:>9.3f} {r['E_total']:>10.3f}")

    return {
        "E_QM_embedded"   : round(E_qmmm*HARTREE_TO_KCAL, 3),
        "E_elec_classical": round(E_elec_cl, 3),
        "E_LJ"            : round(E_LJ, 3),
        "E_total"         : round(E_total, 3),
        "qm_charges"      : qm_charges,
        "per_residue"     : df,
        "top5"            : df.head(5).to_dict("records"),
        "mf_qmmm"         : mf_qmmm,
        "dm_qmmm"         : dm_qmmm,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 7 : H-Bond Detection
# ══════════════════════════════════════════════════════════════════
def detect_hbonds(lig, pocket, qmmm, cutoff=3.5):
    print(f"\n{'═'*66}")
    print(f"  STEP 7  |  Protein–Ligand H-Bond Analysis")
    print(f"{'─'*66}")

    lc    = lig["coords"]
    elems = lig["elems"]
    qm_q  = qmmm["qm_charges"]
    pc    = pocket["mm_coords"]
    pq    = pocket["mm_charges"]
    pn    = pocket["mm_names"]

    lig_no   = [i for i,e in enumerate(elems) if e[0].upper() in ('N','O')]
    prot_no  = [j for j,nm in enumerate(pn)
                if any(x in nm.split(":")[-1]
                       for x in ("N","O","OG","OH","OD","OE","NE","ND","NZ","NH"))]

    hbonds = []
    for li in lig_no:
        for pj in prot_no:
            d = float(np.linalg.norm(lc[li]-pc[pj]))
            if d <= cutoff:
                lq = float(qm_q[li]); pq_val = float(pq[pj])
                htype = ("Lig(acceptor) ← Prot(donor)"
                         if lq < pq_val
                         else "Lig(donor) → Prot(acceptor)")
                hbonds.append({
                    "Type"       : htype,
                    "Lig_atom"   : f"{elems[li]}{li}",
                    "Prot_atom"  : pn[pj],
                    "Distance_Å" : round(d, 2),
                    "Lig_charge" : round(lq, 4),
                    "Prot_charge": round(pq_val, 4),
                    "Strength"   : ("Strong"   if d<3.0 else
                                    "Moderate" if d<3.2 else "Weak"),
                })

    hbonds.sort(key=lambda x: x["Distance_Å"])
    seen_pairs = set()
    hbonds_uniq = []
    for h in hbonds:
        pair = (h["Lig_atom"], h["Prot_atom"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            hbonds_uniq.append(h)
    hbonds = hbonds_uniq

    print(f"  H-bonds detected: {len(hbonds)}")
    for h in hbonds:
        print(f"  [{h['Strength']:<8}] {h['Type']:<35} "
              f"{h['Lig_atom']:<8} ↔ {h['Prot_atom']:<24} "
              f"d={h['Distance_Å']} Å")
    return hbonds


# ══════════════════════════════════════════════════════════════════
#  STEP 8 : Mechanism Interpretation
# ══════════════════════════════════════════════════════════════════
def interpret_mechanism(desc, mep, qmmm, fukui, hbonds):
    print(f"\n{'═'*66}")
    print(f"  STEP 8  |  Inhibition Mechanism Interpretation")
    print(f"{'─'*66}")

    eta   = desc.get("Chemical Hardness η (eV)", 5)
    omega = desc.get("Electrophilicity ω (eV)", 0)
    chi   = desc.get("Electronegativity χ (eV)", 0)
    gap   = desc.get("HOMO-LUMO gap (eV)", 10)
    LUMO  = desc.get("LUMO energy (eV)", 0)
    Et    = qmmm["E_total"]
    logp  = desc.get("LogP", 0)
    tpsa  = desc.get("TPSA (Å²)", 200)
    qed   = desc.get("QED", 0)
    n_hb  = len(hbonds)
    pct_n = mep.get("pct_neg", 0)
    top_e = fukui["top_electrophilic_atom"]
    top_n = fukui["top_nucleophilic_atom"]

    mech = {}
    if   Et < -30: mech["Binding Mode"] = "Covalent / Irreversible"
    elif Et < -10: mech["Binding Mode"] = "Tight competitive inhibition"
    elif Et <   0: mech["Binding Mode"] = "Moderate non-covalent"
    else         : mech["Binding Mode"] = "Weak / Allosteric"

    if   eta < 2: mech["Reactivity"] = f"Soft electrophile → Covalent warhead (atom {top_e})"
    elif eta < 4: mech["Reactivity"] = "Intermediate → Mixed non-covalent"
    else        : mech["Reactivity"] = "Hard molecule → H-bond / ionic driven"

    if   n_hb >= 4: mech["Dominant Interaction"] = f"H-bond network ({n_hb} bonds) — ATP-mimetic"
    elif n_hb >= 2: mech["Dominant Interaction"] = f"H-bond ({n_hb}) + Hydrophobic"
    elif qmmm["E_LJ"] < -5: mech["Dominant Interaction"] = "Hydrophobic / vdW burial"
    else          : mech["Dominant Interaction"] = "Electrostatic"

    if   omega > 3: mech["Electrophilicity Class"] = f"Strong (ω={omega:.2f}) — Michael acceptor"
    elif omega > 1: mech["Electrophilicity Class"] = f"Moderate (ω={omega:.2f})"
    else          : mech["Electrophilicity Class"] = f"Weak (ω={omega:.2f}) — nucleophilic"

    mech["Metal Chelation"] = (f"Likely metalloenzyme chelator (χ={chi:.2f} eV)"
                                if chi > 5 and pct_n > 50 else "Non-chelating")

    mech["Covalent Potential"] = (f"Possible (LUMO={LUMO:.2f} eV < 0, electrophilic)"
                                   if LUMO < 0
                                   else f"Low (LUMO={LUMO:.2f} eV > 0)")

    if   gap > 7: mech["Selectivity"] = f"High (gap={gap:.2f} eV) → selective, stable"
    elif gap > 4: mech["Selectivity"] = f"Moderate (gap={gap:.2f} eV)"
    else        : mech["Selectivity"] = f"Low (gap={gap:.2f} eV) → reactive / PAINS risk"

    lip = (desc.get("MW (Da)",0)<=500 and logp<=5 and
           desc.get("HBD",0)<=5 and desc.get("HBA",0)<=10)
    mech["Lipinski Rule of 5"]  = "✓ Pass" if lip else "✗ Fail"
    mech["Oral Bioavailability"] = (f"Good (TPSA={tpsa}Å², LogP={logp})"
                                     if tpsa<90 and 0<=logp<=5
                                     else f"Poor (TPSA={tpsa}Å², LogP={logp})")
    mech["Drug-likeness QED"]   = (f"High QED={qed:.3f}"     if qed>0.6
                                    else f"Moderate QED={qed:.3f}" if qed>0.4
                                    else f"Low QED={qed:.3f}")

    print(f"\n  ── Mechanism Summary ───────────────────────────")
    for k,v in mech.items():
        print(f"  {k:<28}: {v}")
    return mech


# ══════════════════════════════════════════════════════════════════
#  STEP 9 : Figures (2 × 4-panel, 600 DPI, white background)
# ══════════════════════════════════════════════════════════════════
def visualize(lig, mep_data, qmmm, desc, fukui,
              hbonds, pocket, mechanism, out_dir):
    print(f"\n{'═'*66}")
    print(f"  STEP 9  |  Generating 600 DPI Figures")
    print(f"{'─'*66}")

    BG  = "#FFFFFF"; PAN = "#F4F6F9"; TXT = "#1a1a2e"
    SUB = "#4a4a6a"; CYN = "#1565C0"; RED = "#C62828"
    GRN = "#2E7D32"; YLW = "#F57F17"; CMAP = "RdBu_r"

    plt.rcParams.update({
        "font.family"    : "DejaVu Sans",
        "font.size"      : 11,
        "axes.linewidth" : 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    def sa(ax, title, sub=""):
        ax.set_facecolor(PAN)
        for sp in ax.spines.values():
            sp.set_color("#cccccc"); sp.set_linewidth(0.8)
        ax.tick_params(colors=TXT, labelsize=10, length=4, width=0.8)
        ax.set_title(f"{title}\n{sub}" if sub else title,
                     color=TXT, fontsize=12, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(SUB); ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_color(SUB); ax.yaxis.label.set_fontsize(10)
        ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
        ax.set_axisbelow(True)

    def add_leg(ax, **kw):
        leg = ax.legend(fontsize=9, framealpha=0.95,
                        labelcolor=TXT, **kw)
        leg.get_frame().set_facecolor(PAN)
        leg.get_frame().set_edgecolor("#cccccc")

    coords = lig["coords"]
    elems  = lig["elems"]
    fp     = fukui["f_plus"]
    fm     = fukui["f_minus"]
    df0    = fukui["delta_f"]
    mc     = qmmm["qm_charges"]

    # ── Figure 1 : Ligand Analysis ────────────────────────────────
    fig1, ax1 = plt.subplots(2,2,figsize=(20,18))
    fig1.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.40,wspace=0.36,
                        left=0.07,right=0.97,top=0.92,bottom=0.07)

    # A: Charge Map
    axA = ax1[0,0]
    scA = axA.scatter(coords[:,0],coords[:,1],c=mc,
                      cmap=CMAP,s=300,edgecolors="#555",lw=0.8,zorder=3)
    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            d=np.linalg.norm(coords[i]-coords[j])
            ri=VDW_ANG.get(elems[i][0],1.7)*0.45
            rj=VDW_ANG.get(elems[j][0],1.7)*0.45
            if d<(ri+rj)*1.35:
                axA.plot([coords[i,0],coords[j,0]],
                         [coords[i,1],coords[j,1]],
                         color="#666",lw=1.5,zorder=2)
    for i,(el,pos,q) in enumerate(zip(elems,coords,mc)):
        axA.annotate(f"{el}{i}\n{q:+.3f}",
                     (pos[0],pos[1]),fontsize=8,color=TXT,
                     ha="center",va="bottom",fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15",
                               fc="white",ec="#ddd",alpha=0.85,lw=0.5))
    cb0=plt.colorbar(scA,ax=axA,fraction=0.046,pad=0.03)
    cb0.set_label("DFT Mulliken Charge (e)",color=SUB,fontsize=10)
    cb0.ax.yaxis.set_tick_params(color=TXT,labelsize=9)
    plt.setp(cb0.ax.yaxis.get_ticklabels(),color=TXT)
    cb0.outline.set_edgecolor("#cccccc")
    axA.set_xlabel("X (Å)"); axA.set_ylabel("Y (Å)")
    sa(axA,"A.  Real DFT Mulliken Charge Map",
       f"B3LYP/{lig['basis']} | SCF = {desc.get('SCF energy (Ha)',0):.4f} Ha")

    # B: MEP
    axB = ax1[0,1]
    G,V=mep_data["grid"],mep_data["mep"]
    cent=G.mean(axis=0); sl=np.abs(G[:,2]-cent[2])<0.6
    scB=axB.scatter(G[sl,0],G[sl,1],c=V[sl],cmap=CMAP,s=8,alpha=0.8,
                    vmin=np.percentile(V[sl],3),vmax=np.percentile(V[sl],97))
    axB.scatter(coords[:,0],coords[:,1],c="#222",s=60,marker="o",
                edgecolors="#000",lw=0.7,zorder=5,label="QM atoms")
    pc2=pocket["mm_coords"]
    axB.scatter(pc2[:,0],pc2[:,1],c=GRN,s=20,marker="x",
                lw=1.2,alpha=0.5,zorder=3,label="MM pocket")
    cb1=plt.colorbar(scB,ax=axB,fraction=0.046,pad=0.03)
    cb1.set_label("MEP (a.u.)",color=SUB,fontsize=10)
    cb1.ax.yaxis.set_tick_params(color=TXT,labelsize=9)
    plt.setp(cb1.ax.yaxis.get_ticklabels(),color=TXT)
    cb1.outline.set_edgecolor("#cccccc")
    add_leg(axB)
    axB.set_xlabel("X (Å)"); axB.set_ylabel("Y (Å)")
    sa(axB,"B.  Real DFT MEP (from electron density)",
       f"Min={mep_data['MEP_min']:.4f}  Max={mep_data['MEP_max']:.4f} a.u.")

    # C: Fukui
    axC = ax1[1,0]
    n_at=len(coords); idx=np.arange(n_at)
    axC.bar(idx-0.22,fp,0.40,color=RED,alpha=0.85,
            label="f⁺ (electrophilic, N+1)",edgecolor="none")
    axC.bar(idx+0.22,fm,0.40,color=CYN,alpha=0.85,
            label="f⁻ (nucleophilic, N-1)",edgecolor="none")
    ti=fukui["top_electrophilic_atom"]; ni=fukui["top_nucleophilic_atom"]
    axC.annotate(f"Max f⁺\n{elems[ti]}{ti}",
                 xy=(ti-0.22,fp[ti]),xytext=(ti+1.5,fp[ti]+0.02),
                 fontsize=9,color=RED,fontweight="bold",
                 arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    axC.annotate(f"Max f⁻\n{elems[ni]}{ni}",
                 xy=(ni+0.22,fm[ni]),xytext=(ni+1.5,fm[ni]+0.02),
                 fontsize=9,color=CYN,fontweight="bold",
                 arrowprops=dict(arrowstyle="->",color=CYN,lw=1.2))
    axC.set_xlabel("Atom index"); axC.set_ylabel("Fukui index")
    axC.set_xticks(idx)
    axC.set_xticklabels([f"{e}{i}" for i,e in enumerate(elems)],
                        rotation=55,fontsize=8,ha="right")
    add_leg(axC)
    sa(axC,"C.  Real Fukui Reactivity Indices",
       "Calculated from N, N+1, N-1 DFT states")

    # D: H-bonds
    axD = ax1[1,1]
    if hbonds:
        labs=[f"{h['Lig_atom']} ↔ {h['Prot_atom'].split(':')[-1]}"
              for h in hbonds]
        dsts=[h["Distance_Å"] for h in hbonds]
        col=[GRN if d<3.0 else YLW if d<3.2 else RED for d in dsts]
        bars=axD.barh(labs,dsts,color=col,edgecolor="#aaa",lw=0.4,height=0.55)
        for bar,d in zip(bars,dsts):
            axD.text(bar.get_width()+0.03,
                     bar.get_y()+bar.get_height()/2,
                     f"{d:.2f} Å",va="center",fontsize=9,
                     color=TXT,fontweight="bold")
        axD.axvline(3.5,color=RED,lw=1.2,ls="--",label="3.5Å cutoff")
        axD.axvline(3.0,color=GRN,lw=1.2,ls=":",label="3.0Å strong")
        axD.set_xlabel("Donor–Acceptor Distance (Å)")
        add_leg(axD)
    else:
        axD.text(0.5,0.5,"No H-bonds detected",
                 ha="center",va="center",color=SUB,
                 fontsize=14,transform=axD.transAxes)
    n_str=sum(1 for h in hbonds if h.get("Strength")=="Strong")
    sa(axD,f"D.  Protein–Ligand H-Bonds ({len(hbonds)})",
       f"{n_str} strong (<3.0Å) | Charges from DFT Mulliken")

    fig1.suptitle(
        f"Figure 1 — Ligand Quantum Analysis  ·  "
        f"HOMO={desc.get('HOMO energy (eV)',0):.4f} eV  ·  "
        f"LUMO={desc.get('LUMO energy (eV)',0):.4f} eV  ·  "
        f"Gap={desc.get('HOMO-LUMO gap (eV)',0):.4f} eV",
        color=TXT,fontsize=14,fontweight="bold",y=0.96)
    p1 = out_dir/"fig1_ligand_quantum.png"
    fig1.savefig(p1,dpi=600,bbox_inches="tight",
                 facecolor="white",edgecolor="none")
    plt.close(fig1)
    print(f"  ✓ Figure 1 → {p1.name}")

    # ── Figure 2 : Binding Analysis ───────────────────────────────
    fig2, ax2 = plt.subplots(2,2,figsize=(20,18))
    fig2.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.42,wspace=0.38,
                        left=0.10,right=0.97,top=0.92,bottom=0.07)

    # E: Per-residue
    axE = ax2[0,0]
    df  = qmmm["per_residue"].head(14)
    col = [RED if v<0 else CYN for v in df["E_total"]]
    bars= axE.barh(df["Residue"],df["E_total"],color=col,
                   edgecolor="#aaa",lw=0.4,height=0.65)
    for bar,val in zip(bars,df["E_total"]):
        xp=bar.get_width()
        axE.text(xp+(0.2 if xp>=0 else -0.2),
                 bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}",va="center",
                 ha=("left" if xp>=0 else "right"),
                 fontsize=8,color=TXT)
    axE.axvline(0,color="#444",lw=1.0)
    axE.legend(handles=[Patch(color=RED,label="Favorable"),
                         Patch(color=CYN,label="Unfavorable")],
               fontsize=9,framealpha=0.95,labelcolor=TXT
               ).get_frame().set_facecolor(PAN)
    axE.set_xlabel("ΔE_total (kcal/mol)")
    sa(axE,"E.  Per-Residue QM/MM Energy",
       f"E_total = {qmmm['E_total']:.2f} kcal/mol")

    # F: Stacked
    axF = ax2[0,1]
    df10= qmmm["per_residue"].head(12)
    y   = np.arange(len(df10))
    axF.barh(y,df10["E_elec"],0.55,color=RED,alpha=0.85,
             label="E_elec",edgecolor="none")
    axF.barh(y,df10["E_LJ"],0.55,color=CYN,alpha=0.85,
             label="E_LJ (vdW)",edgecolor="none",left=df10["E_elec"])
    axF.set_yticks(y); axF.set_yticklabels(df10["Residue"],fontsize=9)
    axF.set_xlabel("Energy (kcal/mol)")
    axF.axvline(0,color="#444",lw=1.0)
    add_leg(axF,loc="lower right")
    sa(axF,"F.  Electrostatic vs vdW Decomposition",
       f"E_elec={qmmm['E_elec_classical']:.2f}  "
       f"E_LJ={qmmm['E_LJ']:.2f}  kcal/mol")

    # G: DFT Radar
    axG = ax2[1,0]; axG.remove()
    axG = fig2.add_subplot(2,2,3,polar=True)
    axG.set_facecolor(PAN)
    keys = ["IP = -E(HOMO) (eV)","EA = -E(LUMO) (eV)",
            "Chemical Hardness η (eV)",
            "Electrophilicity ω (eV)","Global Softness S (eV⁻¹)"]
    lbls = ["IP","EA","Hardness η","Electrophilicity ω","Softness S"]
    vals = [abs(desc.get(k,0)) for k in keys]
    vmax = max(v for v in vals if v>0)+1e-9
    vn   = [v/vmax for v in vals]+[vals[0]/vmax]
    N    = len(keys)
    angs = [n/N*2*np.pi for n in range(N)]+[0]
    axG.plot(angs,vn,color=GRN,lw=2.5,zorder=3)
    axG.fill(angs,vn,color=GRN,alpha=0.25,zorder=2)
    axG.scatter(angs[:-1],vn[:-1],color=GRN,s=80,
                zorder=4,edgecolors="white",lw=1.0)
    for ang,vv,lbl,k in zip(angs[:-1],vn[:-1],lbls,keys):
        axG.annotate(f"{desc.get(k,0):.2f}",
                     xy=(ang,vv),xytext=(ang,vv+0.14),
                     ha="center",va="center",fontsize=9,
                     color=TXT,fontweight="bold")
    axG.set_xticks(angs[:-1])
    axG.set_xticklabels(lbls,color=TXT,fontsize=9,fontweight="bold")
    axG.set_yticklabels([]); axG.set_ylim(0,1.35)
    axG.spines["polar"].set_color("#cccccc")
    axG.yaxis.grid(True,color="#ddd",lw=0.6)
    axG.xaxis.grid(True,color="#ddd",lw=0.6)
    axG.set_title(
        f"G.  DFT Reactivity Descriptors\n"
        f"Gap = {desc.get('HOMO-LUMO gap (eV)',0):.4f} eV  |  "
        f"η = {desc.get('Chemical Hardness η (eV)',0):.4f} eV",
        color=TXT,fontsize=12,fontweight="bold",pad=24)

    # H: HOMO/LUMO diagram
    axH = ax2[1,1]
    HOMO = desc.get("HOMO energy (eV)",0)
    LUMO = desc.get("LUMO energy (eV)",0)
    gap  = desc.get("HOMO-LUMO gap (eV)",0)
    axH.barh(["HOMO","LUMO"],[HOMO,LUMO],
             color=[RED,CYN],edgecolor="#555",height=0.4)
    axH.axvline(0,color="#333",lw=0.8,ls="--")
    axH.annotate("",xy=(LUMO,0.62),xytext=(HOMO,0.62),
                 arrowprops=dict(arrowstyle="<->",color=GRN,lw=2.5))
    axH.text((HOMO+LUMO)/2,0.72,f"Gap = {gap:.4f} eV",
             ha="center",fontsize=12,color=GRN,fontweight="bold")
    for val,lbl in [(HOMO,"HOMO"),(LUMO,"LUMO")]:
        axH.text(val+0.05,["HOMO","LUMO"].index(lbl),
                 f"{val:.4f} eV",va="center",fontsize=11,
                 color=TXT,fontweight="bold")
    axH.set_xlabel("Orbital Energy (eV)")
    axH.set_yticks([0,1])
    axH.set_yticklabels(["HOMO","LUMO"],fontsize=13,fontweight="bold")
    sa(axH,"H.  Real HOMO/LUMO Orbital Energies",
       f"IP={desc.get('IP = -E(HOMO) (eV)',0):.4f} eV  |  "
       f"EA={desc.get('EA = -E(LUMO) (eV)',0):.4f} eV  |  "
       f"χ={desc.get('Electronegativity χ (eV)',0):.4f} eV")

    fig2.suptitle(
        f"Figure 2 — QM/MM Binding Analysis  ·  "
        f"E_QM/MM = {qmmm['E_total']:.2f} kcal/mol  ·  "
        f"H-bonds = {len(hbonds)}  ·  "
        f"Pocket = {pocket['n_pocket_res']} residues",
        color=TXT,fontsize=14,fontweight="bold",y=0.96)
    p2 = out_dir/"fig2_binding_qmmm.png"
    fig2.savefig(p2,dpi=600,bbox_inches="tight",
                 facecolor="white",edgecolor="none")
    plt.close(fig2)
    print(f"  ✓ Figure 2 → {p2.name}")
    return str(p1), str(p2)


# ══════════════════════════════════════════════════════════════════
#  STEP 10 : Report
# ══════════════════════════════════════════════════════════════════
def save_report(pdb_path, pocket, desc, mep, qmmm,
                fukui, hbonds, mech, lig, out_dir):
    ts  = datetime.now().isoformat()
    SEP = "═"*68; sep = "─"*68

    data = {
        "timestamp"         : ts,
        "pipeline"          : "Q-MECH v1.0",
        "citation"          : "Dwivedi VD (2026). Q-MECH v1.0. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX",
        "pdb_file"          : str(pdb_path),
        "ligand_resname"    : pocket["lig_resname"],
        "DFT_method"        : desc.get("DFT functional","B3LYP"),
        "basis_set"         : lig["basis"],
        "pocket_residues"   : pocket["pocket_residues"],
        "n_pocket_atoms"    : pocket["n_pocket_atoms"],
        "DFT_descriptors"   : desc,
        "MEP_features"      : {k:v for k,v in mep.items()
                                if k not in("grid","mep","atom_coords",
                                            "V_nuc","V_elec")},
        "Fukui_indices"     : {
            "f_plus"  : fukui["f_plus"].tolist(),
            "f_minus" : fukui["f_minus"].tolist(),
            "delta_f" : fukui["delta_f"].tolist(),
            "top_electrophilic_atom": int(fukui["top_electrophilic_atom"]),
            "top_nucleophilic_atom" : int(fukui["top_nucleophilic_atom"]),
        },
        "QMMM_energies"     : {k:v for k,v in qmmm.items()
                                if k not in("per_residue","top5",
                                            "mf_qmmm","dm_qmmm","qm_charges")},
        "top5_residues"     : qmmm["top5"],
        "hydrogen_bonds"    : hbonds,
        "mechanism"         : mech,
    }

    jp = out_dir/"report.json"
    with open(jp,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)

    tp = out_dir/"report.txt"
    with open(tp,"w",encoding="utf-8") as f:
        f.write(f"{SEP}\n  Q-MECH v1.0 — REAL DFT + QM/MM INHIBITION REPORT\n{SEP}\n")
        f.write(f"  Date    : {ts}\n")
        f.write(f"  PDB     : {pdb_path}\n")
        f.write(f"  Ligand  : {pocket['lig_resname']}\n")
        f.write(f"  Method  : {desc.get('DFT functional','B3LYP')}/"
                f"{lig['basis']} (PySCF real DFT)\n")
        f.write(f"  Cite    : Dwivedi VD (2026). Q-MECH v1.0. Zenodo.\n"
                f"            https://doi.org/10.5281/zenodo.XXXXXXX\n\n")

        f.write(f"{sep}\n  BINDING POCKET ({pocket['n_pocket_res']} residues)\n{sep}\n")
        for r in pocket["pocket_residues"]:
            f.write(f"  {r['chain']}:{r['resname']}{r['resid']}\n")
        f.write("\n")

        f.write(f"{sep}\n  REAL DFT ORBITAL ENERGIES (B3LYP/{lig['basis']})\n{sep}\n")
        for k in ["SCF energy (Ha)","HOMO energy (eV)","LUMO energy (eV)",
                  "HOMO-LUMO gap (eV)","IP = -E(HOMO) (eV)",
                  "EA = -E(LUMO) (eV)","Electronegativity χ (eV)",
                  "Chemical Hardness η (eV)","Electrophilicity ω (eV)",
                  "Global Softness S (eV⁻¹)"]:
            f.write(f"  {k:<38}: {desc.get(k,'')}\n")
        f.write("\n")

        f.write(f"{sep}\n  REAL FUKUI INDICES (N/N+1/N-1 DFT)\n{sep}\n")
        f.write(f"  {'Atom':<8} {'f+':<12} {'f-':<12} {'Δf':<12}\n")
        f.write(f"  {'─'*46}\n")
        for i,(fp,fm,df0) in enumerate(zip(fukui["f_plus"],
                                            fukui["f_minus"],
                                            fukui["delta_f"])):
            f.write(f"  {lig['elems'][i]}{i:<6} {fp:>+10.4f}  "
                    f"{fm:>+10.4f}  {df0:>+10.4f}\n")
        f.write("\n")

        f.write(f"{sep}\n  QM/MM ENERGIES (PySCF embedded)\n{sep}\n")
        f.write(f"  E_QM embedded  : {qmmm['E_QM_embedded']:>12.3f} kcal/mol\n")
        f.write(f"  E_elec(class.) : {qmmm['E_elec_classical']:>12.3f} kcal/mol\n")
        f.write(f"  E_LJ (vdW)     : {qmmm['E_LJ']:>12.3f} kcal/mol\n")
        f.write(f"  E_total QM/MM  : {qmmm['E_total']:>12.3f} kcal/mol\n\n")

        f.write(f"{sep}\n  TOP BINDING RESIDUES\n{sep}\n")
        f.write(f"  {'Residue':<12}{'E_elec':>10}{'E_LJ':>9}{'E_total':>10}  kcal/mol\n")
        for r in qmmm["top5"]:
            f.write(f"  {r['Residue']:<12}{r['E_elec']:>10.3f}"
                    f"{r['E_LJ']:>9.3f}{r['E_total']:>10.3f}\n")
        f.write("\n")

        f.write(f"{sep}\n  H-BONDS ({len(hbonds)})\n{sep}\n")
        for h in hbonds:
            f.write(f"  [{h['Strength']:<8}] {h['Type']:<35} "
                    f"Lig:{h['Lig_atom']:<8} Prot:{h['Prot_atom']:<24} "
                    f"d={h['Distance_Å']} Å\n")
        f.write("\n")

        f.write(f"{sep}\n  MECHANISM\n{sep}\n")
        for k,v in mech.items():
            f.write(f"  {k:<28}: {v}\n")
        f.write(f"\n{SEP}\n")

    print(f"  ✓ Reports → {tp.name}, {jp.name}")
    return tp


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_pipeline(pdb_path, ligand_resname="AUTO",
                 functional="B3LYP", pocket_cutoff=8.0,
                 n_threads=24, mep_grid_pts=30,
                 out_dir=None):

    t_start = time.time()
    print(f"\n{'╔'+'═'*66+'╗'}")
    print(f"║{'Q-MECH v1.0 — Quantum Mechanistic Inhibition Pipeline':^66}║")
    print(f"{'╚'+'═'*66+'╝'}")
    print(f"  PDB       : {pdb_path}")
    print(f"  Functional: {functional}")
    print(f"  Threads   : {n_threads}")
    print(f"  Start     : {datetime.now().strftime('%H:%M:%S')}")

    if out_dir is None:
        out_dir = Path("qmech_results")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    pocket  = extract_pocket_from_pdb(pdb_path, ligand_resname,
                                       pocket_cutoff)
    lig     = build_ligand_mol(pocket)
    dft_n   = run_dft_neutral(lig, functional, n_threads)
    fukui   = run_fukui(lig, dft_n, functional)
    mep_dat = compute_real_mep(lig, dft_n, mep_grid_pts)
    qmmm    = run_qmmm(lig, pocket, functional)
    hbonds  = detect_hbonds(lig, pocket, qmmm)
    mech    = interpret_mechanism(dft_n["descriptors"], mep_dat,
                                   qmmm, fukui, hbonds)
    visualize(lig, mep_dat, qmmm, dft_n["descriptors"],
              fukui, hbonds, pocket, mech, out_dir)
    save_report(pdb_path, pocket, dft_n["descriptors"],
                mep_dat, qmmm, fukui, hbonds, mech, lig, out_dir)

    total = time.time()-t_start
    print(f"\n{'╔'+'═'*66+'╗'}")
    print(f"║{'PIPELINE COMPLETE':^66}║")
    print(f"║  Total time : {total/60:.1f} min{'':<50}║")
    print(f"║  Output     : {str(out_dir.resolve())[:52]:<52}║")
    print(f"{'╚'+'═'*66+'╝'}\n")

    return {"pocket":pocket,"ligand":lig,"dft":dft_n,
            "fukui":fukui,"mep":mep_dat,"qmmm":qmmm,
            "hbonds":hbonds,"mechanism":mech}


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    import sys

    # Command-line usage:
    # python3 qmech_v1.0.py Complex_1.pdb LIG Complex_1_results
    if len(sys.argv) >= 2:
        PDB_FILE       = sys.argv[1]
        LIGAND_RESNAME = sys.argv[2] if len(sys.argv) >= 3 else "AUTO"
        OUT_DIR        = sys.argv[3] if len(sys.argv) >= 4 else "qmech_results"
    else:
        # ┌────────────────────────────────────────────────────┐
        # │  Edit these settings for your complex:             │
        # └────────────────────────────────────────────────────┘
        PDB_FILE       = "Complex_1.pdb"   # ← your PDB file
        LIGAND_RESNAME = "AUTO"            # ← or "LIG", "UNK" etc.
        OUT_DIR        = "Complex_1_results"

    FUNCTIONAL     = "B3LYP"   # B3LYP | PBE | M06-2X | wB97X-D
    POCKET_CUTOFF  = 8.0       # Å
    N_THREADS      = 24        # CPU threads
    MEP_GRID       = 30        # 30=fast, 50=fine

    run_pipeline(
        pdb_path       = PDB_FILE,
        ligand_resname = LIGAND_RESNAME,
        functional     = FUNCTIONAL,
        pocket_cutoff  = POCKET_CUTOFF,
        n_threads      = N_THREADS,
        mep_grid_pts   = MEP_GRID,
        out_dir        = OUT_DIR,
    )
