#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Q-MECH v1.0 — Batch Runner                                     ║
# ║  Processes multiple protein-ligand complexes sequentially        ║
# ║                                                                  ║
# ║  Usage: bash batch_run_qmech.sh                                 ║
# ║  Citation: Dwivedi VD (2026). Q-MECH v1.0. Zenodo.             ║
# ║            https://doi.org/10.5281/zenodo.XXXXXXX               ║
# ╚══════════════════════════════════════════════════════════════════╝

SCRIPT="qmech_v1.0.py"
CUDA_DEV=0

# ┌──────────────────────────────────────────────────────┐
# │  Edit: add your PDB files and ligand resnames        │
# └──────────────────────────────────────────────────────┘
PDBS=(
    "Complex_1.pdb"
    "Complex_2.pdb"
    "Complex_3.pdb"
    "Complex_4.pdb"
)

LIGANDS=(
    "AUTO"    # Complex_1 ligand resname (AUTO = auto-detect)
    "AUTO"    # Complex_2
    "AUTO"    # Complex_3
    "AUTO"    # Complex_4
)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Q-MECH v1.0 — BATCH RUN                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Total complexes : ${#PDBS[@]}                                        ║"
echo "║  Start time      : $(date '+%H:%M:%S %d-%b-%Y')                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0; FAIL=0; FAILED_LIST=""

for i in "${!PDBS[@]}"; do
    PDB="${PDBS[$i]}"
    LIG="${LIGANDS[$i]}"
    NAME="${PDB%.pdb}"
    OUTDIR="${NAME}_results"
    LOG="${NAME}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Complex    : $PDB  (Ligand: $LIG)"
    echo "  Output dir : $OUTDIR/"
    echo "  Log file   : $LOG"
    echo "  Start      : $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$PDB" ]; then
        echo "  ✗ ERROR: $PDB not found — skipping"
        FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $PDB"; continue
    fi

    # Run Q-MECH with arguments (script file never modified)
    CUDA_VISIBLE_DEVICES=$CUDA_DEV python3 -u "$SCRIPT" \
        "$PDB" "$LIG" "$OUTDIR" 2>&1 | tee "$LOG"

    if grep -q "PIPELINE COMPLETE" "$LOG"; then
        echo "  ✓ $NAME — COMPLETE"
        PASS=$((PASS+1))
    else
        echo "  ✗ $NAME — FAILED (check $LOG)"
        FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $PDB"
    fi
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BATCH COMPLETE                                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Passed  : $PASS / ${#PDBS[@]}                                           ║"
echo "║  Failed  : $FAIL / ${#PDBS[@]}                                           ║"
echo "║  End time: $(date '+%H:%M:%S %d-%b-%Y')                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

[ -n "$FAILED_LIST" ] && echo "" && echo "  Failed: $FAILED_LIST"

# Summary table
echo ""
echo "  Generating summary table..."
python3 - << 'PYEOF'
import json
from pathlib import Path

names = ["Complex_1","Complex_2","Complex_3","Complex_4"]

SEP = "="*105
print(f"\n{SEP}")
print(f"  Q-MECH v1.0 — RESULTS SUMMARY")
print(SEP)
print(f"  {'Complex':<16} {'HOMO':>8} {'LUMO':>8} {'Gap':>7} "
      f"{'IP':>7} {'EA':>7} {'omega':>8} {'eta':>7} "
      f"{'QED':>6} {'HB':>5} {'Binding Mode'}")
print(SEP)

for name in names:
    rp = Path(f"{name}_results/report.json")
    if rp.exists():
        with open(rp) as f:
            d = json.load(f)
        desc = d.get("DFT_descriptors", {})
        mech = d.get("mechanism", {})
        hb   = d.get("hydrogen_bonds", [])
        print(f"  {name:<16} "
              f"{desc.get('HOMO energy (eV)','N/A'):>8} "
              f"{desc.get('LUMO energy (eV)','N/A'):>8} "
              f"{desc.get('HOMO-LUMO gap (eV)','N/A'):>7} "
              f"{desc.get('IP = -E(HOMO) (eV)','N/A'):>7} "
              f"{desc.get('EA = -E(LUMO) (eV)','N/A'):>7} "
              f"{desc.get('Electrophilicity ω (eV)','N/A'):>8} "
              f"{desc.get('Chemical Hardness η (eV)','N/A'):>7} "
              f"{desc.get('QED','N/A'):>6} "
              f"{len(hb):>5} "
              f"{mech.get('Binding Mode','N/A')}")
    else:
        print(f"  {name:<16} {'Results not found':>90}")
print(SEP)
PYEOF
