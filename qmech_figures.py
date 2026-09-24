"""
Q-MECH v1.0 — Figure Generator
Generates 8 separate publication-quality figures per complex
+ 1 comparative figure across all complexes

Usage:
    python3 qmech_figures.py

Edit NAMES and LABELS below to match your complex names.

Citation: Dwivedi VD (2026). Q-MECH v1.0. Zenodo.
          https://doi.org/10.5281/zenodo.XXXXXXX
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────
BG  = "#FFFFFF"; PAN = "#F4F6F9"; TXT = "#1a1a2e"
SUB = "#4a4a6a"; CYN = "#1565C0"; RED = "#C62828"
GRN = "#2E7D32"; YLW = "#F57F17"; PRP = "#6A1B9A"
CMAP= "RdBu_r"

plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 12,
    "axes.linewidth" : 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ┌──────────────────────────────────────────────────────────────┐
# │  Edit these to match your complex names                      │
# └──────────────────────────────────────────────────────────────┘
NAMES  = ["Complex_1","Complex_2","Complex_3","Complex_4"]
COLORS = [RED, CYN, GRN, PRP]
LABELS = ["Complex 1","Complex 2","Complex 3","Complex 4"]

VDW_ANG = {'C':1.70,'N':1.55,'O':1.52,'S':1.80,
            'P':1.80,'H':1.20,'F':1.47}

# Load JSON reports
data = {}
for name in NAMES:
    rp = Path(f"{name}_results/report.json")
    if rp.exists():
        with open(rp) as f:
            data[name] = json.load(f)
        print(f"  ✓ Loaded: {name}")
    else:
        print(f"  ✗ Not found: {name}_results/report.json")

OUT = Path("all_figures")
OUT.mkdir(exist_ok=True)

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

def save_fig(fig, path, title=""):
    if title:
        fig.suptitle(title, color=TXT, fontsize=14,
                     fontweight="bold", y=0.98)
    fig.savefig(path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {Path(path).name}")


# ══════════════════════════════════════════════════════════════════
#  Per-complex figures (8 each)
# ══════════════════════════════════════════════════════════════════
for name, col, lbl in zip(NAMES, COLORS, LABELS):
    if name not in data:
        continue
    d    = data[name]
    desc = d.get("DFT_descriptors", {})
    mech = d.get("mechanism", {})
    qmmm = d.get("QMMM_energies", {})
    hb   = d.get("hydrogen_bonds", [])
    top5 = d.get("top5_residues", [])
    fuk  = d.get("Fukui_indices", {})
    mep  = d.get("MEP_features", {})
    short= name.replace(" ","_")
    print(f"\n  {name}")

    fp  = np.array(fuk.get("f_plus", []))
    fm  = np.array(fuk.get("f_minus", []))
    df0 = np.array(fuk.get("delta_f", []))

    # Fig 1: HOMO/LUMO
    fig, ax = plt.subplots(figsize=(10,8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
    HOMO = desc.get("HOMO energy (eV)",0)
    LUMO = desc.get("LUMO energy (eV)",0)
    gap  = desc.get("HOMO-LUMO gap (eV)",0)
    ax.barh(["HOMO","LUMO"],[HOMO,LUMO],
            color=[RED,CYN],edgecolor="#555",height=0.35)
    ax.axvline(0,color="#333",lw=0.8,ls="--")
    ax.annotate("",xy=(LUMO,0.62),xytext=(HOMO,0.62),
                arrowprops=dict(arrowstyle="<->",color=GRN,lw=2.5))
    ax.text((HOMO+LUMO)/2,0.72,f"Gap = {gap:.4f} eV",
            ha="center",fontsize=13,color=GRN,fontweight="bold")
    for val,label in [(HOMO,"HOMO"),(LUMO,"LUMO")]:
        ax.text(val+0.05,["HOMO","LUMO"].index(label),
                f"{val:.4f} eV",va="center",fontsize=12,
                color=TXT,fontweight="bold")
    ax.set_xlabel("Orbital Energy (eV)")
    ax.set_yticks([0,1])
    ax.set_yticklabels(["HOMO","LUMO"],fontsize=13,fontweight="bold")
    for sp in ax.spines.values(): sp.set_color("#cccccc")
    ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
    save_fig(fig, OUT/f"{short}_fig1_homo_lumo.png",
             f"HOMO/LUMO Orbital Energies — {lbl}\n"
             f"B3LYP/{desc.get('Basis set','def2-SVP')}  |  "
             f"IP={desc.get('IP = -E(HOMO) (eV)',0):.4f} eV  |  "
             f"EA={desc.get('EA = -E(LUMO) (eV)',0):.4f} eV")

    # Fig 2: DFT Radar
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
    keys = ["IP = -E(HOMO) (eV)","EA = -E(LUMO) (eV)",
            "Chemical Hardness η (eV)",
            "Electrophilicity ω (eV)","Global Softness S (eV⁻¹)"]
    lbls = ["IP","EA","Hardness η","Electrophilicity ω","Softness S"]
    vals = [abs(desc.get(k,0)) for k in keys]
    vmax = max(v for v in vals if v>0)+1e-9
    vn   = [v/vmax for v in vals]+[vals[0]/vmax]
    N    = len(keys)
    angs = [n/N*2*np.pi for n in range(N)]+[0]
    ax.plot(angs,vn,color=col,lw=2.5,zorder=3)
    ax.fill(angs,vn,color=col,alpha=0.25,zorder=2)
    ax.scatter(angs[:-1],vn[:-1],color=col,s=100,
               zorder=4,edgecolors="white",lw=1.5)
    for ang,vv,l,k in zip(angs[:-1],vn[:-1],lbls,keys):
        ax.annotate(f"{desc.get(k,0):.3f}",
                    xy=(ang,vv),xytext=(ang,vv+0.13),
                    ha="center",va="center",fontsize=11,
                    color=TXT,fontweight="bold")
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(lbls,color=TXT,fontsize=11,fontweight="bold")
    ax.set_yticklabels([]); ax.set_ylim(0,1.35)
    ax.spines["polar"].set_color("#cccccc")
    ax.yaxis.grid(True,color="#ddd",lw=0.6)
    ax.xaxis.grid(True,color="#ddd",lw=0.6)
    save_fig(fig, OUT/f"{short}_fig2_dft_radar.png",
             f"DFT Global Reactivity Descriptors — {lbl}")

    # Fig 3: Fukui
    if len(fp) > 0:
        fig, ax = plt.subplots(figsize=(16,7))
        fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
        idx = np.arange(len(fp))
        ax.bar(idx-0.25,fp,0.22,color=RED,alpha=0.85,
               label="f⁺ (electrophilic, N+1)",edgecolor="none")
        ax.bar(idx,     fm,0.22,color=CYN,alpha=0.85,
               label="f⁻ (nucleophilic, N-1)",edgecolor="none")
        ax.bar(idx+0.25,df0,0.22,color=GRN,alpha=0.85,
               label="Δf (dual descriptor)",edgecolor="none")
        ax.axhline(0,color="#333",lw=0.8)
        ti = int(fuk.get("top_electrophilic_atom",0))
        ni = int(fuk.get("top_nucleophilic_atom",0))
        if len(fp)>ti:
            ax.annotate(f"Max f⁺\natom {ti}",
                        xy=(ti-0.25,fp[ti]),xytext=(ti+2,max(fp)*0.9),
                        fontsize=10,color=RED,fontweight="bold",
                        arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
        if len(fm)>ni:
            ax.annotate(f"Max f⁻\natom {ni}",
                        xy=(ni,fm[ni]),xytext=(ni+2,max(fm)*0.85),
                        fontsize=10,color=CYN,fontweight="bold",
                        arrowprops=dict(arrowstyle="->",color=CYN,lw=1.2))
        ax.set_xlabel("Atom Index"); ax.set_ylabel("Fukui Index")
        add_leg(ax)
        for sp in ax.spines.values(): sp.set_color("#cccccc")
        ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
        ax.set_axisbelow(True)
        save_fig(fig, OUT/f"{short}_fig3_fukui.png",
                 f"Fukui Reactivity Indices — {lbl}\n"
                 f"N/N+1/N-1 DFT B3LYP/{desc.get('Basis set','def2-SVP')}")

    # Fig 4: MEP features
    fig, ax = plt.subplots(figsize=(10,7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
    mep_keys = ["MEP_min","MEP_max","MEP_mean","MEP_std"]
    mep_vals = [mep.get(k,0) for k in mep_keys]
    mep_lbls = ["V_min\n(nucleophilic)","V_max\n(electrophilic)",
                "V_mean","V_std"]
    bar_cols = [CYN,RED,YLW,GRN]
    bars = ax.bar(mep_lbls,mep_vals,color=bar_cols,
                  edgecolor="#aaa",linewidth=0.5,width=0.55)
    for bar,val in zip(bars,mep_vals):
        rng = max(mep_vals)-min(mep_vals)
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(rng*0.02 if rng>0 else 0.001),
                f"{val:.4f}",ha="center",fontsize=11,
                color=TXT,fontweight="bold")
    ax.axhline(0,color="#333",lw=0.8,ls="--")
    ax.set_ylabel("MEP (a.u.)")
    for sp in ax.spines.values(): sp.set_color("#cccccc")
    ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, OUT/f"{short}_fig4_mep.png",
             f"MEP Surface Features — {lbl}\n"
             f"% Positive: {mep.get('pct_pos',0):.1f}%  |  "
             f"% Negative: {mep.get('pct_neg',0):.1f}%")

    # Fig 5: Per-residue energy
    if top5:
        fig, ax = plt.subplots(figsize=(12,8))
        fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
        res  = [r["Residue"] for r in top5]
        etot = [r["E_total"] for r in top5]
        eel  = [r["E_elec"]  for r in top5]
        elj  = [r["E_LJ"]    for r in top5]
        y    = np.arange(len(res))
        ax.barh(y,eel,0.35,color=RED,alpha=0.85,
                label="E_elec",edgecolor="none")
        ax.barh(y,elj,0.35,color=CYN,alpha=0.85,
                label="E_LJ (vdW)",edgecolor="none",left=eel)
        ax.set_yticks(y)
        ax.set_yticklabels(res,fontsize=12,fontweight="bold")
        ax.axvline(0,color="#333",lw=0.8)
        for i,(et,el,lj) in enumerate(zip(etot,eel,elj)):
            ax.text(min(et,0)-0.5,i,f"{et:.3f}",
                    va="center",ha="right",fontsize=10,color=TXT)
        ax.set_xlabel("Energy (kcal/mol)")
        add_leg(ax)
        for sp in ax.spines.values(): sp.set_color("#cccccc")
        ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
        ax.set_axisbelow(True)
        save_fig(fig, OUT/f"{short}_fig5_per_residue.png",
                 f"Top Binding Residues QM/MM Energy — {lbl}\n"
                 f"E_total QM/MM = {qmmm.get('E_total',0):.3f} kcal/mol")

    # Fig 6: Elec vs LJ
    if top5:
        fig, ax = plt.subplots(figsize=(12,8))
        fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
        xp = np.arange(len(res)); ww = 0.35
        ax.bar(xp-ww/2,eel,ww,color=RED,alpha=0.85,
               label="E_electrostatic",edgecolor="none")
        ax.bar(xp+ww/2,elj,ww,color=CYN,alpha=0.85,
               label="E_LJ (vdW)",edgecolor="none")
        ax.set_xticks(xp)
        ax.set_xticklabels(res,rotation=35,ha="right",
                           fontsize=11,fontweight="bold")
        ax.axhline(0,color="#333",lw=0.8)
        ax.set_ylabel("Energy (kcal/mol)")
        add_leg(ax)
        for sp in ax.spines.values(): sp.set_color("#cccccc")
        ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
        ax.set_axisbelow(True)
        save_fig(fig, OUT/f"{short}_fig6_elec_vs_lj.png",
                 f"Electrostatic vs vdW Decomposition — {lbl}\n"
                 f"E_elec={qmmm.get('E_elec_classical',0):.3f}  "
                 f"E_LJ={qmmm.get('E_LJ',0):.3f}  kcal/mol")

    # Fig 7: H-bonds
    fig, ax = plt.subplots(figsize=(14,max(6,len(hb)*0.6+2)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
    if hb:
        labs=[f"{h['Lig_atom']} ↔ {h['Prot_atom'].split(':')[-1]}"
              for h in hb]
        dsts=[h["Distance_Å"] for h in hb]
        colors=[GRN if d<3.0 else YLW if d<3.2 else RED for d in dsts]
        bars=ax.barh(labs,dsts,color=colors,
                     edgecolor="#aaa",lw=0.4,height=0.6)
        for bar,d in zip(bars,dsts):
            ax.text(bar.get_width()+0.02,
                    bar.get_y()+bar.get_height()/2,
                    f"{d:.2f} Å",va="center",fontsize=10,
                    color=TXT,fontweight="bold")
        ax.axvline(3.5,color=RED,lw=1.2,ls="--",label="3.5Å cutoff")
        ax.axvline(3.0,color=GRN,lw=1.2,ls=":",label="3.0Å strong")
        ax.axvline(3.2,color=YLW,lw=1.2,ls="-.",label="3.2Å moderate")
        ax.set_xlabel("Donor–Acceptor Distance (Å)")
        ax.set_xlim(0,max(dsts)+0.8)
        add_leg(ax)
        n_str=sum(1 for h in hb if h.get("Strength")=="Strong")
        n_mod=sum(1 for h in hb if h.get("Strength")=="Moderate")
        n_wk =sum(1 for h in hb if h.get("Strength")=="Weak")
    else:
        ax.text(0.5,0.5,"No H-bonds detected",
                ha="center",va="center",color=SUB,
                fontsize=14,transform=ax.transAxes)
        n_str=n_mod=n_wk=0
    for sp in ax.spines.values(): sp.set_color("#cccccc")
    ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, OUT/f"{short}_fig7_hbonds.png",
             f"Protein–Ligand H-Bonds — {lbl}  ({len(hb)} total)\n"
             f"Strong(<3.0Å):{n_str}  Moderate(3.0-3.2Å):{n_mod}  "
             f"Weak(3.2-3.5Å):{n_wk}")

    # Fig 8: Mechanism summary
    fig, ax = plt.subplots(figsize=(13,9))
    fig.patch.set_facecolor(BG); ax.axis("off")
    rows = list(mech.items())
    y_start=0.91; dy=0.082
    ax.text(0.5,0.97,f"Inhibition Mechanism — {lbl}",
            ha="center",va="center",fontsize=15,
            fontweight="bold",color=TXT,
            transform=ax.transAxes)
    ax.plot([0,1],[0.94,0.94],color="#cccccc",lw=1.0,
            transform=ax.transAxes,clip_on=False)
    for i,(k,v) in enumerate(rows):
        y = y_start - i*dy
        bg = "#EEF2FF" if i%2==0 else PAN
        ax.add_patch(plt.Rectangle((0,y-0.04),1,dy,
                     facecolor=bg,edgecolor="none",
                     transform=ax.transAxes))
        ax.text(0.02,y,k,ha="left",va="center",
                fontsize=12,color=SUB,fontweight="bold",
                transform=ax.transAxes)
        vc=(GRN if "Pass" in str(v) or "Good" in str(v)
            else RED if "Fail" in str(v) or "Poor" in str(v)
            else TXT)
        ax.text(0.40,y,str(v),ha="left",va="center",
                fontsize=12,color=vc,fontweight="bold",
                transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0,0.01),1.0,0.06,
                 facecolor="#E8F5E9",edgecolor="#cccccc",lw=0.8,
                 transform=ax.transAxes))
    summary=(f"HOMO={desc.get('HOMO energy (eV)',0):.4f} eV  |  "
             f"LUMO={desc.get('LUMO energy (eV)',0):.4f} eV  |  "
             f"Gap={desc.get('HOMO-LUMO gap (eV)',0):.4f} eV  |  "
             f"ω={desc.get('Electrophilicity ω (eV)',0):.4f} eV  |  "
             f"QED={desc.get('QED',0):.3f}  |  "
             f"H-bonds={len(hb)}")
    ax.text(0.5,0.04,summary,ha="center",va="center",
            fontsize=10,color=TXT,fontweight="bold",
            transform=ax.transAxes)
    save_fig(fig, OUT/f"{short}_fig8_mechanism.png")


# ══════════════════════════════════════════════════════════════════
#  Comparison Figure — All complexes (6 panels)
# ══════════════════════════════════════════════════════════════════
print("\n  Generating comparison figure...")

fig, axes = plt.subplots(2,3,figsize=(24,16))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.42,wspace=0.35,
                    left=0.07,right=0.97,top=0.91,bottom=0.07)

x = np.arange(len(NAMES)); w = 0.35

def sa_comp(ax, title, sub=""):
    ax.set_facecolor(PAN)
    for sp in ax.spines.values():
        sp.set_color("#cccccc"); sp.set_linewidth(0.8)
    ax.tick_params(colors=TXT,labelsize=10,length=4,width=0.8)
    ax.set_title(f"{title}\n{sub}" if sub else title,
                 color=TXT,fontsize=12,fontweight="bold",pad=8)
    ax.xaxis.label.set_color(SUB); ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_color(SUB); ax.yaxis.label.set_fontsize(10)
    ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
    ax.set_axisbelow(True)

def add_leg_c(ax,**kw):
    leg=ax.legend(fontsize=9,framealpha=0.95,labelcolor=TXT,**kw)
    leg.get_frame().set_facecolor(PAN)
    leg.get_frame().set_edgecolor("#cccccc")

homo_vals = [data[n].get("DFT_descriptors",{}).get("HOMO energy (eV)",0)
             for n in NAMES if n in data]
lumo_vals = [data[n].get("DFT_descriptors",{}).get("LUMO energy (eV)",0)
             for n in NAMES if n in data]
gap_vals  = [data[n].get("DFT_descriptors",{}).get("HOMO-LUMO gap (eV)",0)
             for n in NAMES if n in data]
omega_vals= [data[n].get("DFT_descriptors",{}).get("Electrophilicity ω (eV)",0)
             for n in NAMES if n in data]
hb_total  = [len(data[n].get("hydrogen_bonds",[])) for n in NAMES if n in data]
hb_strong = [sum(1 for h in data[n].get("hydrogen_bonds",[])
                 if h.get("Strength")=="Strong") for n in NAMES if n in data]
top_res   = [data[n].get("top5_residues",[{}])[0].get("Residue","")
             for n in NAMES if n in data]
top_e     = [data[n].get("top5_residues",[{}])[0].get("E_total",0)
             for n in NAMES if n in data]
qed_vals  = [data[n].get("DFT_descriptors",{}).get("QED",0)
             for n in NAMES if n in data]
eta_vals  = [data[n].get("DFT_descriptors",{}).get("Chemical Hardness η (eV)",0)
             for n in NAMES if n in data]

# A: HOMO/LUMO
ax=axes[0,0]; ax.set_facecolor(PAN)
ax.bar(x-w/2,homo_vals,w,color=RED,alpha=0.85,label="HOMO",edgecolor="none")
ax.bar(x+w/2,lumo_vals,w,color=CYN,alpha=0.85,label="LUMO",edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(LABELS,rotation=20,ha="right",fontsize=10)
ax.axhline(0,color="#333",lw=0.8,ls="--")
ax.set_ylabel("Energy (eV)")
add_leg_c(ax)
sa_comp(ax,"A.  HOMO / LUMO Energies","B3LYP/def2-SVP")

# B: Gap
ax=axes[0,1]; ax.set_facecolor(PAN)
bars=ax.bar(LABELS,gap_vals,color=COLORS,alpha=0.85,edgecolor="#aaa",lw=0.4)
for bar,val in zip(bars,gap_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.03,f"{val:.4f}",
            ha="center",fontsize=9,color=TXT,fontweight="bold")
ax.set_ylabel("Gap (eV)")
ax.set_xticklabels(LABELS,rotation=20,ha="right",fontsize=10)
sa_comp(ax,"B.  HOMO-LUMO Gap","Smaller = more reactive")

# C: ω
ax=axes[0,2]; ax.set_facecolor(PAN)
bars=ax.bar(LABELS,omega_vals,color=COLORS,alpha=0.85,edgecolor="#aaa",lw=0.4)
for bar,val in zip(bars,omega_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.1,f"{val:.3f}",
            ha="center",fontsize=9,color=TXT,fontweight="bold")
ax.set_ylabel("ω (eV)")
ax.set_xticklabels(LABELS,rotation=20,ha="right",fontsize=10)
sa_comp(ax,"C.  Electrophilicity Index (ω)","Higher = stronger electrophile")

# D: H-bonds
ax=axes[1,0]; ax.set_facecolor(PAN)
ax.bar(x-w/2,hb_total, w,color=COLORS,alpha=0.85,label="Total",edgecolor="none")
ax.bar(x+w/2,hb_strong,w,color=COLORS,alpha=0.4,label="Strong (<3.0Å)",
       edgecolor="none",hatch="///")
ax.set_xticks(x); ax.set_xticklabels(LABELS,rotation=20,ha="right",fontsize=10)
ax.set_ylabel("Count")
add_leg_c(ax)
sa_comp(ax,"D.  Hydrogen Bond Analysis","Total vs Strong (<3.0 Å)")

# E: Top residue
ax=axes[1,1]; ax.set_facecolor(PAN)
xlbls=[f"{l}\n({r})" for l,r in zip(LABELS,top_res)]
bars=ax.bar(xlbls,top_e,color=COLORS,alpha=0.85,edgecolor="#aaa",lw=0.4)
for bar,val in zip(bars,top_e):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()-0.3,f"{val:.2f}",
            ha="center",fontsize=9,color="white",fontweight="bold")
ax.set_ylabel("E_total (kcal/mol)")
ax.set_xticklabels(xlbls,rotation=0,fontsize=9)
sa_comp(ax,"E.  Strongest Binding Residue","Top QM/MM interaction")

# F: QED vs η
ax=axes[1,2]; ax.set_facecolor(PAN)
ax2=ax.twinx()
ax.bar(x-w/2,qed_vals,w,color=COLORS,alpha=0.85,label="QED",edgecolor="none")
ax2.bar(x+w/2,eta_vals,w,color=COLORS,alpha=0.4,label="η (eV)",
        edgecolor="none",hatch="///")
ax.axhline(0.6,color=GRN,lw=1.0,ls="--",label="QED=0.6")
ax.set_xticks(x)
ax.set_xticklabels(LABELS,rotation=20,ha="right",fontsize=10)
ax.set_ylabel("QED Score",color=TXT,fontsize=10)
ax2.set_ylabel("Chemical Hardness η (eV)",color=SUB,fontsize=10)
ax2.tick_params(colors=TXT,labelsize=9)
for sp in ax.spines.values(): sp.set_color("#cccccc")
l1,lb1=ax.get_legend_handles_labels()
l2,lb2=ax2.get_legend_handles_labels()
leg=ax.legend(l1+l2,lb1+lb2,fontsize=9,framealpha=0.95,labelcolor=TXT)
leg.get_frame().set_facecolor(PAN); leg.get_frame().set_edgecolor("#cccccc")
sa_comp(ax,"F.  Drug-likeness (QED) vs Hardness (η)","Higher QED = drug-like")

fig.suptitle(
    "Q-MECH v1.0 — Comparative DFT + QM/MM Analysis\n"
    "B3LYP/def2-SVP | PySCF + GPU4PySCF",
    color=TXT,fontsize=14,fontweight="bold",y=0.97)
out_path = OUT/"comparison_all_complexes.png"
fig.savefig(out_path,dpi=600,bbox_inches="tight",
            facecolor="white",edgecolor="none")
plt.close(fig)
print(f"  ✓ comparison_all_complexes.png")

print(f"\n  Total figures: {len(list(OUT.glob('*.png')))}")
print(f"  Output folder: {OUT.resolve()}")
print("\n  Q-MECH v1.0 figure generation complete!")
