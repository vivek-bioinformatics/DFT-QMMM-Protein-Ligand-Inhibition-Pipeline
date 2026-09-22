#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Batch DFT + QM/MM Pipeline                                     ║
# ║  Runs each PDB complex sequentially on GPU                      ║
# ║                                                                  ║
# ║  Usage:                                                          ║
# ║    bash batch_run.sh                                             ║
# ║                                                                  ║
# ║  Edit PDBS array below with your PDB file names                 ║
# ╚══════════════════════════════════════════════════════════════════╝

SCRIPT="real_dft_qmmm_pipeline.py"   # Main pipeline script
LIGAND="AUTO"                          # AUTO = auto-detect, or set "LIG", "UNK" etc.
CUDA_DEV=0                             # GPU device (0 = first GPU)

# ─────────────────────────────────────────────────────────────────
#  ADD YOUR PDB FILES HERE
# ─────────────────────────────────────────────────────────────────
PDBS=(
    "complex1.pdb"
    "complex2.pdb"
    "complex3.pdb"
    "complex4.pdb"
)
# ─────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         BATCH DFT + QM/MM PIPELINE                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Start : $(date)"
echo "  Total : ${#PDBS[@]} complexes"
echo ""

PASS=0
FAIL=0
FAILED_LIST=""

for PDB in "${PDBS[@]}"; do

    NAME="${PDB%.pdb}"
    OUTDIR="${NAME}_results"
    LOG="${NAME}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Processing : $PDB"
    echo "  Output     : $OUTDIR/"
    echo "  Log        : $LOG"
    echo "  Time       : $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check PDB exists
    if [ ! -f "$PDB" ]; then
        echo "  ✗ ERROR: $PDB not found — skipping"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $PDB"
        continue
    fi

    # Update script settings
    sed -i "s|PDB_FILE.*=.*|PDB_FILE       = \"$PDB\"|" "$SCRIPT"
    sed -i "s|LIGAND_RESNAME.*=.*|LIGAND_RESNAME = \"$LIGAND\"|" "$SCRIPT"
    sed -i "s|OUT = Path.*|OUT = Path(\"$OUTDIR\")|" "$SCRIPT"

    # Run pipeline
    CUDA_VISIBLE_DEVICES=$CUDA_DEV python3 -u "$SCRIPT" 2>&1 | tee "$LOG"

    # Check success
    if grep -q "PIPELINE COMPLETE" "$LOG"; then
        echo ""
        echo "  ✓ $NAME — COMPLETE"
        PASS=$((PASS + 1))
    else
        echo ""
        echo "  ✗ $NAME — FAILED (check $LOG)"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $PDB"
    fi

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BATCH COMPLETE                                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Passed : $PASS / ${#PDBS[@]}                                          ║"
echo "║  Failed : $FAIL / ${#PDBS[@]}                                          ║"
echo "║  End    : $(date)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ -n "$FAILED_LIST" ]; then
    echo "  Failed:$FAILED_LIST"
fi

# ── Summary Table ─────────────────────────────────────────────────
python3 - << 'PYEOF'
import json
from pathlib import Path
import sys

pdbs = [p.strip() for p in open("batch_run.sh").readlines()
        if p.strip().endswith('.pdb"') and "complex" not in p]
names = [p.strip().strip('"').replace('.pdb','') for p in pdbs]

if not names:
    sys.exit(0)

print("\n" + "="*100)
print(f"  {'Complex':<25} {'HOMO(eV)':>10} {'LUMO(eV)':>10} "
      f"{'Gap(eV)':>8} {'omega':>8} {'E_QM/MM':>12} "
      f"{'QED':>6}  Mechanism")
print("="*100)

for name in names:
    rp = Path(f"{name}_results/report.json")
    if rp.exists():
        with open(rp) as f:
            d = json.load(f)
        desc = d.get("DFT_descriptors", {})
        mech = d.get("mechanism", {})
        qmmm = d.get("QMMM_energies", {})
        print(f"  {name:<25} "
              f"{str(desc.get('HOMO energy (eV)','N/A')):>10} "
              f"{str(desc.get('LUMO energy (eV)','N/A')):>10} "
              f"{str(desc.get('HOMO-LUMO gap (eV)','N/A')):>8} "
              f"{str(desc.get('Electrophilicity omega (eV)',desc.get('Electrophilicity ω (eV)','N/A'))):>8} "
              f"{str(qmmm.get('E_total','N/A')):>12} "
              f"{str(desc.get('QED','N/A')):>6}  "
              f"{mech.get('Binding Mode','N/A')}")
    else:
        print(f"  {name:<25} {'Results not found'}")

print("="*100 + "\n")
PYEOF

