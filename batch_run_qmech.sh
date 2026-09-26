#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Q-MECH v1.0 — Batch Runner                                     ║
# ║  Processes multiple protein-ligand complexes sequentially        ║
# ║                                                                  ║
# ║  Citation: Dwivedi VD (2026). Q-MECH v1.0. Zenodo.             ║
# ║            https://doi.org/10.5281/zenodo.XXXXXXX               ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage (no editing needed):
#   bash batch_run_qmech.sh                     # every *.pdb in the current folder
#   bash batch_run_qmech.sh -d /path/to/pdbs    # every *.pdb in another folder
#   bash batch_run_qmech.sh -f complexes.csv    # per-complex ligand / charge
#   bash batch_run_qmech.sh -F -t GyrB          # + combined figures at the end
#   bash batch_run_qmech.sh A.pdb B.pdb         # only these files (options in any order)
#
# Options:
#   -d DIR     folder with the PDB files               (default: current folder)
#   -f FILE    CSV with one complex per line:  pdb,ligand,charge
#              ligand = AUTO | RESNAME | RESNAME:CHAIN:RESID ; charge = AUTO | 0 | -1 | +1 ...
#              empty fields = AUTO ; lines starting with # are ignored
#   -l LIG     ligand for every complex                 (default: AUTO)
#   -c CHARGE  net charge for every complex             (default: AUTO)
#   -g GPU     GPU index, or "cpu" to force CPU          (default: 0)
#   -s SCRIPT  path to qmech_v1.0.py   (default: next to this script, else current folder)
#   -k         skip complexes that already have <name>_results/report.json
#   -F         run combined_figures.py after the batch
#   -t TARGET  protein name for combined_figures.py text (with -F)
#   -h         show this help

DIR="."; CSV=""; DEF_LIG="AUTO"; DEF_CHG="AUTO"; GPU="0"; SCRIPT=""
SKIP_DONE=0; MAKE_FIGS=0; TARGET=""

usage() { sed -n '/^# Usage/,/^#   -h/p' "$0" | sed 's/^# \{0,1\}//'; }

# options and file names may be given in any order
OPTS=(); FILES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -[dflcgst]) OPTS+=("$1" "$2"); shift 2 ;;
        -[kFh])     OPTS+=("$1"); shift ;;
        -*)         echo "  ✗ Unknown option: $1"; usage; exit 1 ;;
        *)          FILES+=("$1"); shift ;;
    esac
done
set -- "${OPTS[@]}"

while getopts "d:f:l:c:g:s:t:kFh" opt; do
    case $opt in
        d) DIR="$OPTARG" ;;       f) CSV="$OPTARG" ;;
        l) DEF_LIG="$OPTARG" ;;   c) DEF_CHG="$OPTARG" ;;
        g) GPU="$OPTARG" ;;       s) SCRIPT="$OPTARG" ;;
        t) TARGET="$OPTARG" ;;    k) SKIP_DONE=1 ;;
        F) MAKE_FIGS=1 ;;         h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND-1))

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$SCRIPT" ]; then
    if   [ -f "$HERE/qmech_v1.0.py" ]; then SCRIPT="$HERE/qmech_v1.0.py"
    elif [ -f "./qmech_v1.0.py" ];     then SCRIPT="$(pwd)/qmech_v1.0.py"
    fi
fi
[ -n "$SCRIPT" ] && [ -f "$SCRIPT" ] || { echo "  ✗ qmech_v1.0.py not found — use -s /path/to/qmech_v1.0.py"; exit 1; }
SCRIPT="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
FIGS="$(dirname "$SCRIPT")/combined_figures.py"
[ -n "$CSV" ] && CSV="$(cd "$(dirname "$CSV")" && pwd)/$(basename "$CSV")"

cd "$DIR" || { echo "  ✗ Folder not found: $DIR"; exit 1; }

# ── Build the list of complexes ─────────────────────────────────────
PDBS=(); LIGANDS=(); CHARGES=()
if [ -n "$CSV" ]; then
    [ -f "$CSV" ] || { echo "  ✗ CSV not found: $CSV"; exit 1; }
    while IFS=, read -r p l c || [ -n "$p" ]; do
        p="$(echo "$p" | xargs)"; l="$(echo "$l" | xargs)"; c="$(echo "$c" | xargs)"
        [ -z "$p" ] && continue
        case "$p" in \#*|pdb|PDB) continue ;; esac
        PDBS+=("$p"); LIGANDS+=("${l:-$DEF_LIG}"); CHARGES+=("${c:-$DEF_CHG}")
    done < "$CSV"
elif [ ${#FILES[@]} -gt 0 ]; then
    for p in "${FILES[@]}"; do PDBS+=("$p"); LIGANDS+=("$DEF_LIG"); CHARGES+=("$DEF_CHG"); done
else
    shopt -s nullglob
    for p in *.pdb *.PDB; do PDBS+=("$p"); LIGANDS+=("$DEF_LIG"); CHARGES+=("$DEF_CHG"); done
    shopt -u nullglob
fi
[ ${#PDBS[@]} -gt 0 ] || { echo "  ✗ No PDB files found in $(pwd)"; exit 1; }

if [ "$GPU" = "cpu" ] || [ "$GPU" = "CPU" ]; then GPU_ENV=""; GPU_TXT="CPU (forced)"
else GPU_ENV="$GPU"; GPU_TXT="GPU $GPU (CPU if no GPU found)"; fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Q-MECH v1.0 — BATCH RUN                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-16s: %-42s║\n" "Total complexes" "${#PDBS[@]}"
printf "║  %-16s: %-42s║\n" "Folder" "$(pwd | tail -c 42)"
printf "║  %-16s: %-42s║\n" "Device" "$GPU_TXT"
printf "║  %-16s: %-42s║\n" "Start time" "$(date '+%H:%M:%S %d-%b-%Y')"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0; WARN=0; FAIL=0; SKIP=0; FAILED_LIST=""; NAMES=()

for i in "${!PDBS[@]}"; do
    PDB="${PDBS[$i]}"; LIG="${LIGANDS[$i]}"; CHG="${CHARGES[$i]}"
    BASE="$(basename "$PDB")"; NAME="${BASE%.*}"
    OUTDIR="${NAME}_results"; LOG="${NAME}.log"
    NAMES+=("$NAME")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Complex    : $PDB  (Ligand: $LIG, charge: $CHG)   [$((i+1))/${#PDBS[@]}]"
    echo "  Output dir : $OUTDIR/"
    echo "  Log file   : $LOG"
    echo "  Start      : $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$PDB" ]; then
        echo "  ✗ ERROR: $PDB not found — skipping"
        FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $PDB"; echo ""; continue
    fi
    if [ $SKIP_DONE -eq 1 ] && [ -f "$OUTDIR/report.json" ]; then
        echo "  ↷ $NAME — already done (-k), skipped"
        SKIP=$((SKIP+1)); echo ""; continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ENV python3 -u "$SCRIPT" \
        "$PDB" "$LIG" "$OUTDIR" "$CHG" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}

    case $RC in
        0) echo "  ✓ $NAME — COMPLETE"; PASS=$((PASS+1)) ;;
        2) echo "  ⚠ $NAME — COMPLETE, but an SCF did not converge (check $LOG)"
           WARN=$((WARN+1)) ;;
        *) echo "  ✗ $NAME — FAILED (exit $RC, check $LOG)"
           FAIL=$((FAIL+1)); FAILED_LIST="$FAILED_LIST $PDB" ;;
    esac
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BATCH COMPLETE                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-16s: %-42s║\n" "Completed" "$PASS / ${#PDBS[@]}"
printf "║  %-16s: %-42s║\n" "SCF warnings" "$WARN"
printf "║  %-16s: %-42s║\n" "Skipped (-k)" "$SKIP"
printf "║  %-16s: %-42s║\n" "Failed" "$FAIL"
printf "║  %-16s: %-42s║\n" "End time" "$(date '+%H:%M:%S %d-%b-%Y')"
echo "╚══════════════════════════════════════════════════════════════╝"
[ -n "$FAILED_LIST" ] && echo "" && echo "  Failed:$FAILED_LIST"

# ── Summary table (+ CSV) ───────────────────────────────────────────
echo ""
echo "  Generating summary table..."
python3 - "${NAMES[@]}" << 'PYEOF'
import json, sys, csv
from pathlib import Path

names = sys.argv[1:]
rows = []
for name in names:
    rp = Path(f"{name}_results/report.json")
    if not rp.exists():
        rows.append({"Complex": name, "Status": "no results"}); continue
    d = json.loads(rp.read_text(encoding="utf-8"))
    ds, e = d.get("DFT_descriptors", {}), d.get("QMMM_energies", {})
    r2 = lambda v, n=2: "" if v is None else round(float(v), n)
    ok = ds.get("SCF converged") and d.get("Fukui_indices", {}).get("converged", True) \
         and e.get("converged", True)
    rows.append({
        "Complex": name,
        "Ligand": d.get("ligand_resname", ""),
        "q": d.get("ligand_net_charge", ""),
        "Solvent": "PCM" if "PCM" in str(ds.get("Solvent model", "")) else "gas",
        "HOMO": r2(ds.get("HOMO energy (eV)")), "LUMO": r2(ds.get("LUMO energy (eV)")),
        "Gap": r2(ds.get("HOMO-LUMO gap (eV)")), "eta": r2(ds.get("Chemical Hardness η (eV)")),
        "omega": r2(ds.get("Electrophilicity ω (eV)")), "QED": r2(ds.get("QED"), 3),
        "HB": len(d.get("hydrogen_bonds", [])),
        "E_elec": r2(e.get("E_elec_classical"), 1), "E_pol": r2(e.get("E_pol"), 1),
        "E_LJ": r2(e.get("E_LJ"), 1), "E_total": r2(e.get("E_total"), 1),
        "Binding_Mode": d.get("mechanism", {}).get("Binding Mode", ""),
        "Status": "ok" if ok else "SCF warning",
    })

cols = ["Complex","Ligand","q","Solvent","HOMO","LUMO","Gap","eta","omega","QED","HB",
        "E_elec","E_pol","E_LJ","E_total","Binding_Mode","Status"]
with open("qmech_batch_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)

wid = max([len("Complex")] + [len(str(r["Complex"])) for r in rows]) + 2
show = ["q","Solvent","HOMO","LUMO","Gap","eta","omega","QED","HB","E_LJ","E_total","Status"]
SEP = "=" * (wid + 9*len(show) + 6)
print(f"\n{SEP}\n  Q-MECH v1.0 — RESULTS SUMMARY  (eV; energies in kcal/mol)\n{SEP}")
print("  " + f"{'Complex':<{wid}}" + "".join(f"{c:>9}" for c in show))
print(SEP)
for r in rows:
    print("  " + f"{str(r['Complex']):<{wid}}" + "".join(f"{str(r.get(c,'')):>9}" for c in show))
print(SEP)
print("  E_total = gas-phase QM/MM interaction energy (not ΔG); compare only ligands of equal charge q.")
print("  Saved: qmech_batch_summary.csv")
PYEOF

# ── Optional: combined figures ──────────────────────────────────────
if [ $MAKE_FIGS -eq 1 ]; then
    echo ""
    if [ -f "$FIGS" ]; then
        echo "  Generating combined figures..."
        DONE=(); for n in "${NAMES[@]}"; do [ -f "${n}_results/report.json" ] && DONE+=("$n"); done
        if [ ${#DONE[@]} -gt 0 ]; then
            if [ -n "$TARGET" ]; then python3 "$FIGS" --names "${DONE[@]}" --target "$TARGET"
            else                      python3 "$FIGS" --names "${DONE[@]}"; fi
        else
            echo "  ✗ No finished results — figures skipped"
        fi
    else
        echo "  ✗ combined_figures.py not found next to $SCRIPT — figures skipped"
    fi
fi
