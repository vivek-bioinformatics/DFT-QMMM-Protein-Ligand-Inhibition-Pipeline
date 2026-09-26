"""
Q-MECH v1.0 — Combined Figure Generator
8 combined figures (one panel per compound, 2 columns, any number of
compounds) + 1 comparison figure + manuscript-style results text.
Reads <NAME>_results/report.json (and report.txt) written by qmech_v1.0.py.

Usage (run in the folder that contains the *_results directories):
    python3 combined_figures.py                         # all *_results found here
    python3 combined_figures.py --target GyrB           # protein name in the text
    python3 combined_figures.py --names 7PQL_A 7PQL_B 7PQL_Ref \
                                --labels A B "80S (reference)" --target GyrB
    python3 combined_figures.py --dir /path/to/results --out my_figures --dpi 300

Options:
    --names   result folder names without "_results", in the order to plot
              (default: all found; names containing "ref" are placed last)
    --labels  display names (default: folder names minus their common prefix)
    --target  protein name used in the results text (default: "protein")
    --dir     folder with the *_results directories (default: current folder)
    --out     output folder (default: combined_figures)
    --dpi     figure resolution (default: 600)
"""

import json, os, argparse, string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────
BG  = "#FFFFFF"; PAN = "#F4F6F9"; TXT = "#1a1a2e"
SUB = "#4a4a6a"; CYN = "#1565C0"; RED = "#C62828"
GRN = "#2E7D32"; YLW = "#F57F17"; PRP = "#6A1B9A"

plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 10,
    "axes.linewidth" : 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ── Command-line settings (no editing needed) ───────────────────────
ap = argparse.ArgumentParser(description="Q-MECH combined figures and results text")
ap.add_argument("--names",  nargs="+", help="result folder names without _results, in plot order")
ap.add_argument("--labels", nargs="+", help="display names, same order as --names")
ap.add_argument("--target", default="", help="protein name used in the text, e.g. GyrB")
ap.add_argument("--dir",    default=".", help="folder containing the *_results directories")
ap.add_argument("--out",    default="combined_figures", help="output folder")
ap.add_argument("--dpi",    type=int, default=600)
ARGS = ap.parse_args()
os.chdir(ARGS.dir)

if ARGS.names:
    NAMES = [n[:-8] if n.endswith("_results") else n for n in ARGS.names]
else:
    NAMES = sorted(d[:-8] for d in os.listdir(".")
                   if d.endswith("_results") and os.path.isfile(os.path.join(d, "report.json")))
    NAMES = [n for n in NAMES if "ref" not in n.lower()] + \
            [n for n in NAMES if "ref" in n.lower()]          # reference(s) last
if not NAMES:
    raise SystemExit("  No *_results/report.json found — run this script in the folder that "
                     "contains the Q-MECH result directories (or use --dir).")

if ARGS.labels:
    if len(ARGS.labels) != len(NAMES):
        raise SystemExit(f"  --labels has {len(ARGS.labels)} entries but there are "
                         f"{len(NAMES)} compounds.")
    LABELS = list(ARGS.labels)
elif len(NAMES) > 1:
    cp  = os.path.commonprefix(NAMES)
    cut = cp.rfind("_") + 1                                    # strip e.g. "7PQL_"
    LABELS = [n[cut:] or n for n in NAMES]
else:
    LABELS = list(NAMES)

TARGET = ARGS.target.strip()
DPI    = ARGS.dpi
_extra = ["#EF6C00", "#00838F", "#AD1457", "#5D4037", "#455A64", "#9E9D24",
          "#283593", "#00695C", "#6D4C41", "#C2185B", "#512DA8", "#0277BD"]
COLORS = ([RED, CYN, GRN, PRP] + _extra * (len(NAMES) // len(_extra) + 1))[:len(NAMES)]
PANEL  = [string.ascii_uppercase[i] if i < 26 else f"A{i-25}" for i in range(len(NAMES))]
NR     = max(1, -(-len(NAMES) // 2))                        # rows of the 2-column grids

# Load data
data = {}
for name in NAMES:
    rp = Path(f"{name}_results/report.json")
    if rp.exists():
        with open(rp, encoding="utf-8") as f:
            data[name] = json.load(f)
        print(f"  ✓ Loaded: {name}")
    else:
        print(f"  ✗ Not found: {name}_results/report.json")
if not data:
    raise SystemExit("  No report.json found — check NAMES and run this script "
                     "in the folder that contains the *_results directories.")

# only compounds that were found, in the chosen order
ENTRIES = [(n, l, c, p) for n, l, c, p in zip(NAMES, LABELS, COLORS, PANEL) if n in data]
first   = data[ENTRIES[0][0]]
METHOD  = f"{first.get('DFT_method','B3LYP')}/{first.get('basis_set','def2-SVP')}"
SOLVTXT = first.get("DFT_descriptors", {}).get("Solvent model", "gas phase")
if SOLVTXT != "gas phase":
    _sv = SOLVTXT.split("(", 1)[-1].rstrip(")").split(",")[0].strip()
    METHOD_FULL = f"{METHOD} level with the IEF-PCM implicit solvent model ({_sv})"
else:
    METHOD_FULL = f"{METHOD} level (gas phase)"
CHG = lambda n: data[n].get("ligand_net_charge", data[n].get("DFT_descriptors", {}).get("Net charge", 0)) or 0

OUT = Path(ARGS.out)
OUT.mkdir(exist_ok=True)

import re
def atom_labels(name):
    """['C0','N1','O7',...] from the Fukui table in <name>_results/report.txt."""
    rp = Path(f"{name}_results/report.txt")
    labs = []
    if rp.exists():
        txt = rp.read_text(encoding="utf-8")
        sec = txt.split("REAL FUKUI INDICES", 1)[-1].split("QM/MM ENERGIES", 1)[0]
        for line in sec.splitlines():
            mm = re.match(r"\s+([A-Z][a-z]?)(\d+)\s+[+-]\d", line)
            if mm:
                labs.append(mm.group(1).capitalize() + mm.group(2))
    return labs

def lab_of(name, i):
    labs = LABS.get(name, [])
    return labs[i] if i < len(labs) else f"atom {i}"

def num(v, default=np.nan):
    """float or NaN (report values can be None)."""
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def fmt(v, nd=3):
    v = num(v)
    return "N/A" if np.isnan(v) else f"{v:.{nd}f}"

LABS = {n: atom_labels(n) for n, *_ in ENTRIES}

def style_ax(ax):
    ax.set_facecolor(PAN)
    for sp in ax.spines.values():
        sp.set_color("#cccccc"); sp.set_linewidth(0.6)
    ax.tick_params(colors=TXT, labelsize=7, length=3, width=0.6)
    ax.grid(True, color="#e2e2e2", lw=0.4, ls="--", alpha=0.8)
    ax.set_axisbelow(True)

def sa(ax, title, sub=""):
    """Panel style for the comparison figure (was missing in the original)."""
    style_ax(ax)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{title}\n{sub}" if sub else title,
                 color=TXT, fontsize=13, fontweight="bold", pad=10)

def add_leg(ax, **kw):
    kw.setdefault("fontsize", 6)
    leg = ax.legend(framealpha=0.9, labelcolor=TXT, **kw)
    leg.get_frame().set_facecolor(PAN)
    leg.get_frame().set_edgecolor("#cccccc")

def grid_axes(axes):
    """Pair each found compound with an axis; hide unused axes."""
    flat = list(np.ravel(axes))
    for ax in flat[len(ENTRIES):]:
        ax.axis("off")
    return zip(flat, ENTRIES)

def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {Path(path).name}")

print("\n  Generating combined figures...")

# ══════════════════════════════════════════════════════════════════
# Fig 1: HOMO/LUMO — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(14/2.54, 14/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.55, wspace=0.55,
                    left=0.12, right=0.97,
                    top=0.96, bottom=0.08)

for ax, (name, lbl, col, pan) in grid_axes(axes):
    desc = data[name].get("DFT_descriptors", {})
    HOMO = num(desc.get("HOMO energy (eV)"), 0)
    LUMO = num(desc.get("LUMO energy (eV)"), 0)
    gap  = num(desc.get("HOMO-LUMO gap (eV)"), 0)

    style_ax(ax)
    ax.barh(["HOMO","LUMO"], [HOMO, LUMO],
            color=[RED, CYN], edgecolor="#555", height=0.35)
    ax.axvline(0, color="#333", lw=0.6, ls="--")
    ax.annotate("", xy=(LUMO, 0.62), xytext=(HOMO, 0.62),
                arrowprops=dict(arrowstyle="<->", color=GRN, lw=1.5))
    ax.text((HOMO+LUMO)/2, 0.76, f"Gap={gap:.3f} eV",
            ha="center", fontsize=7, color=GRN, fontweight="bold")
    for val, label in [(HOMO,"HOMO"),(LUMO,"LUMO")]:
        ax.text(val/2, ["HOMO","LUMO"].index(label),
                f"{val:.3f} eV", va="center", ha="center",
                fontsize=7, color="white", fontweight="bold")
    ax.set_xlabel("Orbital Energy (eV)", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HOMO","LUMO"],
                       fontsize=8, fontweight="bold")
    ax.set_title(f"({pan}) {lbl}", color=TXT,
                 fontsize=9, fontweight="bold", pad=6)

save(fig, OUT/"combined_fig1_homo_lumo.png")

# ══════════════════════════════════════════════════════════════════
# Fig 2: DFT Radar — all (2×2)
# Each descriptor is scaled by its maximum ACROSS compounds, so the
# radar shapes can be compared between compounds (mixed units otherwise).
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(14/2.54, 14/2.54*NR/2), squeeze=False,
                          subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.55, wspace=0.45,
                    left=0.05, right=0.95,
                    top=0.93, bottom=0.05)

keys = ["IP = -E(HOMO) (eV)","EA = -E(LUMO) (eV)",
        "Chemical Hardness η (eV)",
        "Electrophilicity ω (eV)","Global Softness S (eV⁻¹)"]
rlbls = ["IP","EA","η","ω","S"]
N    = len(keys)
angs = [n/N*2*np.pi for n in range(N)] + [0]
kmax = {k: max(abs(num(data[n]["DFT_descriptors"].get(k), 0)) for n, *_ in ENTRIES) or 1.0
        for k in keys}

for ax, (name, lbl, col, pan) in grid_axes(axes):
    desc = data[name].get("DFT_descriptors", {})
    vn   = [abs(num(desc.get(k), 0))/kmax[k] for k in keys]
    vn   = vn + [vn[0]]

    ax.set_facecolor(PAN)
    ax.plot(angs, vn, color=col, lw=2.0, zorder=3)
    ax.fill(angs, vn, color=col, alpha=0.25, zorder=2)
    ax.scatter(angs[:-1], vn[:-1], color=col, s=50,
               zorder=4, edgecolors="white", lw=1.0)
    for ang, vv, k in zip(angs[:-1], vn[:-1], keys):
        ax.annotate(fmt(desc.get(k), 2),
                    xy=(ang, vv), xytext=(ang, vv+0.18),
                    ha="center", va="center", fontsize=6,
                    color=TXT, fontweight="bold")
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(rlbls, color=TXT,
                       fontsize=7, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.4)
    ax.spines["polar"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#ddd", lw=0.5)
    ax.xaxis.grid(True, color="#ddd", lw=0.5)
    ax.set_title(f"({pan}) {lbl}", color=TXT,
                 fontsize=9, fontweight="bold", pad=18)
fig.text(0.5, 0.985, "Each axis scaled to the largest value among the compounds",
         ha="center", va="top", fontsize=6, color=SUB)

save(fig, OUT/"combined_fig2_dft_radar.png")

# ══════════════════════════════════════════════════════════════════
# Fig 3: Fukui — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(20/2.54, 16/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.60, wspace=0.35,
                    left=0.07, right=0.97,
                    top=0.96, bottom=0.08)

for ax, (name, lbl, col, pan) in grid_axes(axes):
    fuk = data[name].get("Fukui_indices", {})
    fp  = np.array(fuk.get("f_plus",  []), dtype=float)
    fm  = np.array(fuk.get("f_minus", []), dtype=float)
    df0 = np.array(fuk.get("delta_f", []), dtype=float)
    if len(fp) == 0:
        ax.axis("off"); continue

    style_ax(ax)
    idx = np.arange(len(fp))
    ax.bar(idx-0.22, fp,  0.20, color=RED, alpha=0.85,
           label="f⁺", edgecolor="none")
    ax.bar(idx,      fm,  0.20, color=CYN, alpha=0.85,
           label="f⁻", edgecolor="none")
    ax.bar(idx+0.22, df0, 0.20, color=GRN, alpha=0.85,
           label="Δf", edgecolor="none")
    ax.axhline(0, color="#333", lw=0.6)

    ti = int(fuk.get("top_electrophilic_atom", np.argmax(fp)))
    ni = int(fuk.get("top_nucleophilic_atom",  np.argmax(fm)))
    ytop = max(fp.max(), fm.max())
    ax.axvline(ti-0.22, color=RED, lw=1.0, ls=":", alpha=0.7)
    ax.text(ti-0.22, ytop*1.02, f"f⁺max\n{lab_of(name, ti)}",
            ha="center", va="bottom", fontsize=5.5, color=RED, fontweight="bold")
    ax.axvline(ni, color=CYN, lw=1.0, ls=":", alpha=0.7)
    ax.text(ni, ytop*1.02 if ni != ti else ytop*1.25, f"f⁻max\n{lab_of(name, ni)}",
            ha="center", va="bottom", fontsize=5.5, color=CYN, fontweight="bold")
    ax.set_ylim(min(0, df0.min())*1.15, ytop*1.55)

    if LABS.get(name) and len(LABS[name]) == len(fp):
        step = max(1, len(fp)//16)
        ax.set_xticks(idx[::step]); ax.set_xticklabels(LABS[name][::step], rotation=90, fontsize=5)
    ax.set_xlabel("Atom", fontsize=7)
    ax.set_ylabel("Fukui Index (Hirshfeld)", fontsize=7)
    add_leg(ax, ncol=3, loc="upper right")
    ax.set_title(f"({pan}) {lbl}", color=TXT,
                 fontsize=9, fontweight="bold", pad=6)

save(fig, OUT/"combined_fig3_fukui.png")

# ══════════════════════════════════════════════════════════════════
# Fig 4: MEP features — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(14/2.54, 12/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.60, wspace=0.45,
                    left=0.10, right=0.97,
                    top=0.96, bottom=0.10)

for ax, (name, lbl, col, pan) in grid_axes(axes):
    mep      = data[name].get("MEP_features", {})
    mep_keys = ["MEP_min","MEP_max","MEP_mean","MEP_std"]
    mep_vals = [num(mep.get(k), 0) for k in mep_keys]
    mep_lbls = ["V_min","V_max","V_mean","V_std"]
    bar_cols = [CYN, RED, YLW, GRN]

    style_ax(ax)
    bars = ax.bar(mep_lbls, mep_vals, color=bar_cols,
                  edgecolor="#aaa", linewidth=0.4, width=0.6)
    rng  = (max(mep_vals)-min(mep_vals)
            if max(mep_vals) != min(mep_vals) else 0.001)
    for bar, val in zip(bars, mep_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                val + (rng*0.03 if val >= 0 else -rng*0.03),
                f"{val:.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=6,
                color=TXT, fontweight="bold")
    ax.set_ylim(min(mep_vals)-rng*0.18, max(mep_vals)+rng*0.18)
    ax.axhline(0, color="#333", lw=0.6, ls="--")
    ax.set_ylabel("MEP (a.u.)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    near = (f"min near {mep.get('nuc_site_atom','?')}, "
            f"max near {mep.get('elec_site_atom','?')}")
    ax.set_title(f"({pan}) {lbl}\n{near}", color=TXT,
                 fontsize=8, fontweight="bold", pad=5)

save(fig, OUT/"combined_fig4_mep.png")

# ══════════════════════════════════════════════════════════════════
# Fig 5: Per-residue QM/MM energy (E_total) — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(16/2.54, 14/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.60, wspace=0.60,
                    left=0.14, right=0.97,
                    top=0.94, bottom=0.08)

for ax, (name, lbl, col, pan) in grid_axes(axes):
    top5 = data[name].get("top5_residues", [])
    qmmm = data[name].get("QMMM_energies", {})
    if not top5:
        ax.axis("off"); continue

    res  = [r["Residue"] for r in top5]
    etot = [num(r["E_total"], 0) for r in top5]
    y    = np.arange(len(res))

    style_ax(ax)
    ax.barh(y, etot, 0.55, color=[RED if e < 0 else CYN for e in etot],
            alpha=0.85, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(res, fontsize=7, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.6)
    span = max(abs(min(etot + [0])), abs(max(etot + [0]))) or 1
    for i, et in enumerate(etot):
        ax.text(et + (-0.03*span if et < 0 else 0.03*span), i, f"{et:.1f}",
                va="center", ha="right" if et < 0 else "left",
                fontsize=6, color=TXT)
    ax.set_xlim(min(etot + [0]) - 0.25*span, max(etot + [0]) + 0.25*span)
    ax.set_xlabel("E_elec + E_LJ (kcal/mol)", fontsize=7)
    ax.set_title(f"({pan}) {lbl}\nE_total={num(qmmm.get('E_total'),0):.1f} kcal/mol",
                 color=TXT, fontsize=8, fontweight="bold", pad=5)

save(fig, OUT/"combined_fig5_per_residue.png")

# ══════════════════════════════════════════════════════════════════
# Fig 6: Electrostatic vs LJ — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(16/2.54, 14/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.65, wspace=0.45,
                    left=0.10, right=0.97,
                    top=0.96, bottom=0.12)

for ax, (name, lbl, col, pan) in grid_axes(axes):
    top5 = data[name].get("top5_residues", [])
    if not top5:
        ax.axis("off"); continue

    res = [r["Residue"] for r in top5]
    eel = [num(r["E_elec"], 0) for r in top5]
    elj = [num(r["E_LJ"], 0)   for r in top5]
    x   = np.arange(len(res)); w = 0.35

    style_ax(ax)
    ax.bar(x-w/2, eel, w, color=RED, alpha=0.85,
           label="E_elec", edgecolor="none")
    ax.bar(x+w/2, elj, w, color=CYN, alpha=0.85,
           label="E_LJ",   edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(res, rotation=30, ha="right",
                       fontsize=6, fontweight="bold")
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=7)
    add_leg(ax, ncol=2)
    ax.set_title(f"({pan}) {lbl}", color=TXT,
                 fontsize=9, fontweight="bold", pad=6)

save(fig, OUT/"combined_fig6_elec_vs_lj.png")

# ══════════════════════════════════════════════════════════════════
# Fig 7: H-bonds — all (2×2)   (strength from report: Jeffrey 1997)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(18/2.54, 16/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.60, wspace=0.75,
                    left=0.20, right=0.97,
                    top=0.94, bottom=0.08)
SCOL = {"Strong": GRN, "Moderate": YLW, "Weak": RED}

for ax, (name, lbl, col, pan) in grid_axes(axes):
    hb = data[name].get("hydrogen_bonds", [])

    style_ax(ax)
    if hb:
        labs = [f"{h['Lig_atom']}↔{h['Prot_atom']}" for h in hb]
        dsts = [num(h["Distance_Å"], 0) for h in hb]
        clrs = [SCOL.get(h.get("Strength"), SUB) for h in hb]
        bars = ax.barh(labs, dsts, color=clrs,
                       edgecolor="#aaa", lw=0.3, height=0.55)
        for bar, h in zip(bars, hb):
            ang = h.get("Angle_°")
            ax.text(bar.get_width()+0.03,
                    bar.get_y()+bar.get_height()/2,
                    f"{num(h['Distance_Å']):.2f}" + (f", {num(ang):.0f}°" if ang is not None else ""),
                    va="center", fontsize=5.5, color=TXT, fontweight="bold")
        ax.axvline(3.5, color=RED, lw=0.8, ls="--", alpha=0.7)
        ax.axvline(3.2, color=YLW, lw=0.8, ls=":",  alpha=0.7)
        ax.set_xlim(0, max(dsts)+1.0)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_xlabel("D···A Distance (Å)", fontsize=7)
        n_str = sum(1 for h in hb if h.get("Strength")=="Strong")
        n_mod = sum(1 for h in hb if h.get("Strength")=="Moderate")
        n_wk  = sum(1 for h in hb if h.get("Strength")=="Weak")
        subtitle = f"Total:{len(hb)}  Strong:{n_str}  Mod:{n_mod}  Weak:{n_wk}"
    else:
        ax.text(0.5, 0.5, "No H-bonds detected",
                ha="center", va="center", color=SUB,
                fontsize=9, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        subtitle = "No H-bonds"
    ax.set_title(f"({pan}) {lbl}\n{subtitle}",
                 color=TXT, fontsize=8, fontweight="bold", pad=5)
fig.text(0.5, 0.005, "Strength (Jeffrey 1997, D···A): Strong < 2.5 Å, Moderate 2.5–3.2 Å, "
         "Weak > 3.2 Å  |  angle D–H···A ≥ 120°", ha="center", fontsize=5.5, color=SUB)

save(fig, OUT/"combined_fig7_hbonds.png")

# ══════════════════════════════════════════════════════════════════
# Fig 8: Mechanism summary — all (2×2)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(NR, 2, figsize=(18/2.54, 16/2.54*NR/2), squeeze=False)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.10, wspace=0.08,
                    left=0.02, right=0.98,
                    top=0.98, bottom=0.02)

def wrap(s, width):
    words, lines, cur = str(s).split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > width:
            lines.append(cur); cur = w_
        else:
            cur = (cur + " " + w_).strip()
    lines.append(cur)
    return lines

for ax, (name, lbl, col, pan) in grid_axes(axes):
    mech = data[name].get("mechanism", {})
    desc = data[name].get("DFT_descriptors", {})
    hb   = data[name].get("hydrogen_bonds", [])

    ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0), 1, 1,
                 facecolor=PAN, edgecolor=col, lw=1.5,
                 transform=ax.transAxes))
    ax.text(0.5, 0.97, f"({pan}) {lbl}",
            ha="center", va="top", fontsize=9,
            fontweight="bold", color=col,
            transform=ax.transAxes)
    ax.plot([0.02,0.98],[0.925,0.925], color=col,
            lw=0.8, transform=ax.transAxes)

    rows = [(k, wrap(v, 52)[:3]) for k, v in mech.items()]
    n_lines = sum(len(v) for _, v in rows) + 0.4*len(rows)
    lh = min(0.80/max(n_lines, 1), 0.06)
    y  = 0.905
    for i, (k, vlines) in enumerate(rows):
        h  = lh*(len(vlines)+0.4)
        ax.add_patch(plt.Rectangle((0.01, y-h), 0.98, h,
                     facecolor="#FFFFFF" if i%2==0 else "#F0F4FF",
                     edgecolor="none", transform=ax.transAxes))
        ax.text(0.03, y-lh*0.7, k+":", ha="left", va="center", fontsize=5,
                color=SUB, fontweight="bold", transform=ax.transAxes)
        s = str(mech[k])
        vc = (GRN if ("Pass" in s or "Good" in s) else
              RED if ("Fail" in s or "Poor" in s) else TXT)
        for j, t in enumerate(vlines):
            ax.text(0.36, y-lh*(0.7+j), t, ha="left", va="center", fontsize=5,
                    color=vc, transform=ax.transAxes)
        y -= h

    # Bottom summary bar
    ax.add_patch(plt.Rectangle((0.01,0.01), 0.98, 0.06,
                 facecolor="#E8F5E9", edgecolor="none",
                 transform=ax.transAxes))
    summary = (f"HOMO={fmt(desc.get('HOMO energy (eV)'))}  "
               f"LUMO={fmt(desc.get('LUMO energy (eV)'))}  "
               f"ω={fmt(desc.get('Electrophilicity ω (eV)'))}  "
               f"QED={fmt(desc.get('QED'))}  HB={len(hb)}")
    ax.text(0.5, 0.04, summary,
            ha="center", va="center", fontsize=5.5,
            color=TXT, fontweight="bold",
            transform=ax.transAxes)

save(fig, OUT/"combined_fig8_mechanism.png")

print(f"\n  All 8 combined figures complete!")

# ══════════════════════════════════════════════════════════════════
#  Comparison Figure — all complexes
# ══════════════════════════════════════════════════════════════════
print("\n  Generating comparison figure...")

# ══════════════════════════════════════════════════════════════════
#  Comparison Figure — Dr. Dwivedi refined style
#  All fonts black bold | 18×25 cm | 600 DPI
# ══════════════════════════════════════════════════════════════════

# Override rcParams for comparison figure — black bold fonts
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 9,
    "font.weight"     : "bold",
    "axes.linewidth"  : 0.8,
    "axes.titlesize"  : 10,
    "axes.labelsize"  : 9,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize" : 8.5,
    "ytick.labelsize" : 8.5,
    "legend.fontsize" : 8,
    "xtick.direction" : "out",
    "ytick.direction" : "out",
    "text.color"      : "#000000",
    "axes.labelcolor" : "#000000",
    "xtick.color"     : "#000000",
    "ytick.color"     : "#000000",
})

names_f = [n for n, *_ in ENTRIES]
labs_f  = [l for _, l, *_ in ENTRIES]
cols_f  = [c for _, _, c, _ in ENTRIES]
D = lambda n, k: num(data[n].get("DFT_descriptors", {}).get(k), 0)
x = np.arange(len(names_f)); w = 0.35

W_IN = 18.0 / 2.54
H_IN = 25.0 / 2.54

fig, axes = plt.subplots(3, 2, figsize=(W_IN, H_IN))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.55, wspace=0.38,
                    left=0.11, right=0.97,
                    top=0.96, bottom=0.07)

def sa_cmp(ax, title, sub=""):
    ax.set_facecolor(PAN)
    for sp in ax.spines.values():
        sp.set_color("#cccccc"); sp.set_linewidth(0.6)
    ax.tick_params(colors="#000000", labelsize=8.5,
                   length=3, width=0.6, which="both")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("#000000"); lbl.set_fontweight("bold")
    full = f"{title}\n{sub}" if sub else title
    ax.set_title(full, color="#000000", fontsize=10,
                 fontweight="bold", pad=6)
    ax.xaxis.label.set_color("#000000")
    ax.xaxis.label.set_fontsize(9)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_color("#000000")
    ax.yaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontweight("bold")
    ax.grid(True, color="#e2e2e2", lw=0.4, ls="--", alpha=0.8)
    ax.set_axisbelow(True)

def add_leg_cmp(ax, **kw):
    leg = ax.legend(fontsize=8, framealpha=0.95,
                    labelcolor="#000000", **kw)
    leg.get_frame().set_facecolor(PAN)
    leg.get_frame().set_edgecolor("#cccccc")
    for t in leg.get_texts():
        t.set_color("#000000"); t.set_fontweight("bold")

# Panel A: HOMO/LUMO
ax = axes[0, 0]
homo_vals = [D(n, "HOMO energy (eV)") for n in names_f]
lumo_vals = [D(n, "LUMO energy (eV)") for n in names_f]
ax.bar(x-w/2, homo_vals, w, color=RED, alpha=0.85,
       label="HOMO", edgecolor="none")
ax.bar(x+w/2, lumo_vals, w, color=CYN, alpha=0.85,
       label="LUMO", edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(labs_f, rotation=25, ha="right")
ax.axhline(0, color="#333", lw=0.8, ls="--")
ax.set_ylabel("Energy (eV)")
add_leg_cmp(ax, loc="upper right")
sa_cmp(ax, "A.  HOMO / LUMO Energies",
       METHOD + (", IEF-PCM" if SOLVTXT != "gas phase" else ", gas phase"))

# Panel B: HOMO-LUMO gap
ax = axes[0, 1]
gap_vals = [D(n, "HOMO-LUMO gap (eV)") for n in names_f]
bars = ax.bar(labs_f, gap_vals, color=cols_f,
              alpha=0.85, edgecolor="#aaa", lw=0.4)
for bar, val in zip(bars, gap_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.03,
            f"{val:.2f}", ha="center", fontsize=8,
            color="#000000", fontweight="bold")
ax.set_ylabel("Gap (eV)")
ax.set_xticklabels(labs_f, rotation=25, ha="right")
ax.set_ylim(0, max(gap_vals)*1.18)
sa_cmp(ax, "B.  HOMO-LUMO Gap", "Smaller gap = more reactive")

# Panel C: Electrophilicity ω
ax = axes[1, 0]
omega_vals = [D(n, "Electrophilicity ω (eV)") for n in names_f]
bars = ax.bar(labs_f, omega_vals, color=cols_f,
              alpha=0.85, edgecolor="#aaa", lw=0.4)
for bar, val in zip(bars, omega_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.1,
            f"{val:.2f}", ha="center", fontsize=8,
            color="#000000", fontweight="bold")
ax.set_ylabel("ω (eV)")
ax.set_xticklabels(labs_f, rotation=25, ha="right")
ax.set_ylim(0, max(omega_vals)*1.2)
sa_cmp(ax, "C.  Electrophilicity Index (ω)",
       "Higher ω = stronger electrophile")

# Panel D: H-bond count
ax = axes[1, 1]
hb_total  = [len(data[n].get("hydrogen_bonds", [])) for n in names_f]
hb_strong = [sum(1 for h in data[n].get("hydrogen_bonds", [])
                 if h.get("Strength") in ("Strong", "Moderate")) for n in names_f]
ax.bar(x-w/2, hb_total,  w, color=cols_f, alpha=0.85,
       label="Total", edgecolor="none")
ax.bar(x+w/2, hb_strong, w, color=cols_f, alpha=0.4,
       label="Strong + Moderate (≤ 3.2 Å)", edgecolor="none", hatch="///")
ax.set_xticks(x)
ax.set_xticklabels(labs_f, rotation=25, ha="right")
ax.set_ylabel("Count")
ax.yaxis.get_major_locator().set_params(integer=True)
add_leg_cmp(ax)
sa_cmp(ax, "D.  Hydrogen Bond Analysis",
       "Total vs Strong + Moderate (Jeffrey 1997)")

# Panel E: Top residue energy
ax = axes[2, 0]
top_res = [(data[n].get("top5_residues") or [{}])[0].get("Residue", "")
           for n in names_f]
top_e   = [(data[n].get("top5_residues") or [{}])[0].get("E_total", 0)
           for n in names_f]
xlabels = [f"{l}\n({r.split(':')[-1]})" for l, r in zip(labs_f, top_res)]
bars = ax.bar(xlabels, top_e, color=cols_f, alpha=0.85,
              edgecolor="#aaa", lw=0.4)
for bar, val in zip(bars, top_e):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()/2,
            f"{val:.1f}", ha="center", fontsize=8,
            color="white", fontweight="bold")
ax.set_ylabel("E_total (kcal/mol)")
ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=8)
lo_ = min(top_e+[0]); hi_ = max(top_e+[0]); sp_ = (hi_-lo_) or 1
ax.set_ylim(lo_-0.15*sp_, hi_+0.10*sp_)
sa_cmp(ax, "E.  Strongest Binding Residue",
       "Top QM/MM interaction per complex")

# Panel F: QED vs Hardness
ax = axes[2, 1]
qed_vals = [D(n, "QED") for n in names_f]
eta_vals = [D(n, "Chemical Hardness η (eV)") for n in names_f]
ax2 = ax.twinx()
ax.bar(x-w/2, qed_vals, w, color=cols_f, alpha=0.85,
       edgecolor="none", label="QED")
ax2.bar(x+w/2, eta_vals, w, color=cols_f, alpha=0.4,
        edgecolor="none", hatch="///", label="η (eV)")
ax.axhline(0.6, color=GRN, lw=1.0, ls="--", label="QED=0.6")
ax.set_xticks(x)
ax.set_xticklabels(labs_f, rotation=25, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("QED Score", color="#000000",
              fontsize=9, fontweight="bold")
ax2.set_ylabel("Chemical Hardness η (eV)", color="#000000",
               fontsize=9, fontweight="bold")
ax2.tick_params(colors="#000000", labelsize=8.5)
for lbl in ax2.get_yticklabels():
    lbl.set_color("#000000"); lbl.set_fontweight("bold")
sa_cmp(ax, "F.  Drug-likeness (QED) vs Hardness (η)",
       "Higher QED = drug-like | Higher η = less reactive")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax.legend(lines1+lines2, labels1+labels2,
                fontsize=8, framealpha=0.95,
                labelcolor="#000000", loc="upper right")
leg.get_frame().set_facecolor(PAN)
leg.get_frame().set_edgecolor("#cccccc")
for t in leg.get_texts():
    t.set_color("#000000"); t.set_fontweight("bold")

fig.savefig(OUT/"comparison_all_complexes.png",
            dpi=DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("  ✓ comparison_all_complexes.png")


# ══════════════════════════════════════════════════════════════════
#  Results Section (Manuscript draft — every statement from the data)
# ══════════════════════════════════════════════════════════════════
print("\n  Writing results section...")

def omega_class(w):
    """Domingo et al. 2002 global electrophilicity scale."""
    w = num(w)
    if np.isnan(w): return "N/A"
    return ("strong electrophile" if w > 1.5 else
            "moderate electrophile" if w >= 0.8 else "marginal electrophile")

with open(OUT/"results_section.txt","w",encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write("  RESULTS SECTION — DFT + QM/MM ANALYSIS (draft, check before use)\n")
    f.write("="*70 + "\n\n")

    f.write(f"3.X  Quantum Chemical Analysis of {TARGET + ' Inhibitors' if TARGET else 'the Studied Ligands'}\n\n")

    f.write("3.X.1  DFT Electronic Structure and Global Reactivity Descriptors\n\n")
    f.write(f"The electronic structures of {len(names_f)} {TARGET or 'protein'}–ligand complexes were\n")
    f.write(f"investigated at the {METHOD_FULL}\n")
    f.write("using PySCF, with the ligand geometries taken from the complexes\n")
    f.write("(protonation states as prepared; net charges listed in Table X).\n")
    f.write("Frontier orbital energies and\n")
    f.write("conceptual-DFT descriptors (Koopmans approximation: I = −E_HOMO,\n")
    f.write("A = −E_LUMO; χ = (I+A)/2, η = I − A, S = 1/η, ω = χ²/2η; Parr et al.\n")
    f.write("1999) are summarized in Table X.\n\n")

    f.write(f"Table X. DFT Global Reactivity Descriptors ({METHOD}"
            f"{', IEF-PCM' if SOLVTXT != 'gas phase' else ', gas phase'}; q = ligand net charge)\n")
    f.write("-"*90 + "\n")
    f.write(f"  {'Compound':<16} {'q':>3} {'HOMO':>8} {'LUMO':>8} {'Gap':>7} "
            f"{'IP':>7} {'EA':>7} {'chi':>7} {'eta':>7} "
            f"{'omega':>8} {'S':>7}\n")
    f.write(f"  {'':16} {'':>3} {'(eV)':>8} {'(eV)':>8} {'(eV)':>7} "
            f"{'(eV)':>7} {'(eV)':>7} {'(eV)':>7} {'(eV)':>7} "
            f"{'(eV)':>8} {'(eV-1)':>7}\n")
    f.write("-"*90 + "\n")
    for name, lbl, *_ in ENTRIES:
        d = data[name].get("DFT_descriptors",{})
        f.write(f"  {lbl:<16} {CHG(name):>+3d} "
                f"{fmt(d.get('HOMO energy (eV)'),2):>8} "
                f"{fmt(d.get('LUMO energy (eV)'),2):>8} "
                f"{fmt(d.get('HOMO-LUMO gap (eV)'),2):>7} "
                f"{fmt(d.get('IP = -E(HOMO) (eV)'),2):>7} "
                f"{fmt(d.get('EA = -E(LUMO) (eV)'),2):>7} "
                f"{fmt(d.get('Electronegativity χ (eV)'),2):>7} "
                f"{fmt(d.get('Chemical Hardness η (eV)'),2):>7} "
                f"{fmt(d.get('Electrophilicity ω (eV)'),2):>8} "
                f"{fmt(d.get('Global Softness S (eV⁻¹)'),2):>7}\n")
    f.write("-"*90 + "\n\n")

    i_hmax = int(np.argmax(homo_vals)); i_gmin = int(np.argmin(gap_vals))
    i_gmax = int(np.argmax(gap_vals));  i_wmax = int(np.argmax(omega_vals))
    f.write(f"The HOMO energies ranged from {min(homo_vals):.2f} to {max(homo_vals):.2f} eV, "
            f"with {labs_f[i_hmax]} having the highest-lying HOMO (strongest electron donor "
            f"by this criterion). The HOMO–LUMO gaps ranged from {min(gap_vals):.2f} eV "
            f"({labs_f[i_gmin]}) to {max(gap_vals):.2f} eV ({labs_f[i_gmax]}); a smaller gap "
            f"indicates higher chemical reactivity and lower kinetic stability. "
            f"{labs_f[i_wmax]} showed the highest electrophilicity index "
            f"(ω = {max(omega_vals):.2f} eV).\n\n")

    for k_c, (name, lbl, *_) in enumerate(ENTRIES, 1):
        d    = data[name].get("DFT_descriptors",{})
        mech = data[name].get("mechanism",{})
        hb   = data[name].get("hydrogen_bonds",[])
        top5 = data[name].get("top5_residues",[])
        qmmm = data[name].get("QMMM_energies",{})
        fuk  = data[name].get("Fukui_indices",{})
        mep  = data[name].get("MEP_features",{})
        n_str= sum(1 for h in hb if h.get("Strength")=="Strong")
        n_mod= sum(1 for h in hb if h.get("Strength")=="Moderate")
        n_wk = sum(1 for h in hb if h.get("Strength")=="Weak")

        q_txt = f"{CHG(name):+d}" if CHG(name) else "0"
        f.write(f"3.X.1.{k_c}  Compound {lbl} (net charge {q_txt})\n")
        f.write(f"  The HOMO and LUMO energies of {lbl} were {fmt(d.get('HOMO energy (eV)'),2)} "
                f"and {fmt(d.get('LUMO energy (eV)'),2)} eV, giving a HOMO–LUMO gap of "
                f"{fmt(d.get('HOMO-LUMO gap (eV)'),2)} eV (I = {fmt(d.get('IP = -E(HOMO) (eV)'),2)} eV, "
                f"A = {fmt(d.get('EA = -E(LUMO) (eV)'),2)} eV). The chemical hardness was "
                f"η = {fmt(d.get('Chemical Hardness η (eV)'),2)} eV and the electrophilicity index "
                f"ω = {fmt(d.get('Electrophilicity ω (eV)'),2)} eV, which places {lbl} in the "
                f"{omega_class(d.get('Electrophilicity ω (eV)'))} category of the Domingo scale"
                + (" (note: the scale was derived for neutral molecules; ionic species should be "
                   "compared with caution). " if CHG(name) else ". "))
        if fuk.get("f_plus"):
            fp_ = fuk["f_plus"]; fm_ = fuk["f_minus"]
            te, tn = int(fuk.get("top_electrophilic_atom", 0)), int(fuk.get("top_nucleophilic_atom", 0))
            f.write(f"Hirshfeld-condensed Fukui functions identified {lab_of(name, te)} as the most "
                    f"likely site for nucleophilic attack (f⁺ = {fp_[te]:.3f}) and {lab_of(name, tn)} as "
                    f"the most likely site for electrophilic attack (f⁻ = {fm_[tn]:.3f}). ")
        if mep:
            f.write(f"The MEP on the molecular surface ranged from {fmt(mep.get('MEP_min_kcal'),1)} "
                    f"kcal/mol (near {mep.get('nuc_site_atom','?')}) to {fmt(mep.get('MEP_max_kcal'),1)} "
                    f"kcal/mol (near {mep.get('elec_site_atom','?')}).")
        f.write("\n\n")

    f.write("3.X.2  QM/MM Interaction Analysis\n\n")
    for k_c, (name, lbl, *_) in enumerate(ENTRIES, 1):
        mech = data[name].get("mechanism",{})
        hb   = data[name].get("hydrogen_bonds",[])
        top5 = data[name].get("top5_residues",[])
        qmmm = data[name].get("QMMM_energies",{})
        n_str= sum(1 for h in hb if h.get("Strength")=="Strong")
        n_mod= sum(1 for h in hb if h.get("Strength")=="Moderate")
        n_wk = sum(1 for h in hb if h.get("Strength")=="Weak")

        f.write(f"3.X.2.{k_c}  {lbl}\n")
        f.write(f"  Electrostatic-embedding QM/MM calculations gave a ligand–protein "
                f"interaction energy of {fmt(qmmm.get('E_total'),1)} kcal/mol, comprising "
                f"frozen-density electrostatics ({fmt(qmmm.get('E_elec_classical'),1)} kcal/mol), "
                f"ligand polarization ({fmt(qmmm.get('E_pol'),1)} kcal/mol) and van der Waals "
                f"(Lennard-Jones) contributions ({fmt(qmmm.get('E_LJ'),1)} kcal/mol). "
                f"Binding mode: {mech.get('Binding Mode','N/A')}. ")
        if top5:
            f.write(f"The most significant interacting residue was {top5[0]['Residue']} "
                    f"(E = {num(top5[0]['E_total']):.1f} kcal/mol)")
            if len(top5) > 1:
                f.write(f", followed by {top5[1]['Residue']} ({num(top5[1]['E_total']):.1f} kcal/mol)")
            f.write(". ")
        f.write(f"Geometric analysis (D···A ≤ 3.5 Å, ∠D–H···A ≥ 120°) identified {len(hb)} "
                f"protein–ligand hydrogen bond(s): {n_str} strong, {n_mod} moderate and "
                f"{n_wk} weak (Jeffrey classification).\n\n")

    f.write("Note: QM/MM interaction energies are gas-phase values at a fixed geometry\n")
    f.write("(no desolvation, entropy or conformational averaging). Because the binding\n")
    f.write("pocket carries a net charge, these energies are dominated by the ligand net\n")
    f.write("charge and are comparable only between ligands of the same charge; they are\n")
    f.write("not binding free energies. Binding affinities should be ranked with a\n")
    f.write("solvent-screened method (e.g. MM-GBSA).\n\n")

    f.write("="*70 + "\n")
    f.write("  FIGURES LIST\n")
    f.write("="*70 + "\n")
    for p in sorted(OUT.glob("*.png")):
        f.write(f"    {p.name}\n")
    f.write("="*70 + "\n")

print("  ✓ results_section.txt")
print(f"\n  Total figures: {len(list(OUT.glob('*.png')))}")
print(f"  Output folder: {OUT.resolve()}")
