"""
make_matched_tables.py — paper tables and figure from the MATCHED-hyperparameter runs.

Reads only saved outputs; trains nothing. Writes short_paper/matched/ (the older
short_paper/ tables use unequal learning rates and are left untouched).

Matched runs (B and C share one adaptation hyperparameter per label):
  neural  vaso/intubation  seed_floor_lr3e-3.json            (B = C = 3e-3)
          mortality        ../ml4h_clean_v2_mortality/seed_floor.json   (B = C = 3e-4)
  trees   all labels       modular_xgboost_clean_eta0.03.json (eta 0.03 for B and C)
  accuracy inference       A_vs_C_shared.json, pooled_shared.json (in each result dir)
  drift detector           drift_detection_standard.json
  explanation stability    explanation_stability_eta0.03.json, nn_explanation_stability_lr3e-3.json
                           (mortality: ../ml4h_clean_v2_mortality/nn_explanation_stability.json)
  retrieval                retrieval_eta0.03_lr3e-3.json, ../ml4h_clean_v2_mortality/retrieval_eta0.03_bc_states.json

Every B and C mean is cross-checked against the value implied by the inference files
(A + gain); the script stops on any disagreement or mismatch of events, seeds or margin.
Run from ml4h_clean_v2/.
"""
import csv
import json
from pathlib import Path

import numpy as np

V, M = Path("."), Path("../ml4h_clean_v2_mortality")
OUT = Path("short_paper/matched")
OUT.mkdir(parents=True, exist_ok=True)
FINAL = "2020 - 2022"
ERAS = ["2014 - 2016", "2017 - 2019", "2020 - 2022"]
LABELS = [("label_vasopressor", "Vasopressor", "vaso", V), ("label_intubation", "Intubation", "intub", V),
          ("label_mortality", "Mortality", "mort", M)]
FAM = {"neural": "Neural", "trees": "Trees"}


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def jl(p):
    p = Path(p)
    if not p.exists():
        stop(f"missing {p}")
    return json.loads(p.read_text())


# ── load ─────────────────────────────────────────────────────────────────────
nn_run = {V: jl(V / "seed_floor_lr3e-3.json"), M: jl(M / "seed_floor.json")}
tree_run = {V: jl(V / "modular_xgboost_clean_eta0.03.json"), M: jl(M / "modular_xgboost_clean_eta0.03.json")}
avc = {d: jl(d / "A_vs_C_shared.json") for d in (V, M)}
pooled = jl(V / "pooled_shared.json")
drift = jl(V / "drift_detection_standard.json")
MARGIN = nn_run[V]["ni_margin"]
SEEDS = nn_run[V]["seeds"]
for d in (V, M):
    if nn_run[d]["ni_margin"] != MARGIN or tree_run[d]["ni_margin"] != MARGIN or nn_run[d]["seeds"] != SEEDS \
            or tree_run[d]["seeds"] != SEEDS:
        stop(f"margin or seeds differ in {d}")
for d, arm in ((V, "B"), (M, "B")):
    lr = nn_run[d]["best_lr"]
    if lr["B"] != lr["C"]:
        stop(f"neural learning rates are not matched in {d}: {lr}")
    if tree_run[d]["eta"]["B_xgb"] != tree_run[d]["eta"]["C_xgb"]:
        stop(f"tree eta not matched in {d}")


def cell(run, lbl, era):
    for c in run["cells"]:
        if c["label"] == lbl and c["era"] == era:
            return c
    stop(f"no cell {lbl} {era}")


# ── Table 1: accuracy per model family, label and era ────────────────────────
rows = []
for lbl, name, short, d in LABELS:
    for fam in ("neural", "trees"):
        a, b, c_ = ("A", "B", "C") if fam == "neural" else ("A_xgb", "B_xgb", "C_xgb")
        run = nn_run[d] if fam == "neural" else tree_run[d]
        for era in ERAS:
            r = [x for x in avc[d] if x["family"] == fam and x["label"] == name and x["era"] == era]
            if len(r) != 1:
                stop(f"A_vs_C row missing for {fam} {name} {era}")
            r = r[0]
            sc = cell(run, lbl, era)
            if sc["events"] != r["events"]:
                stop(f"event counts differ at {fam} {name} {era}")
            psB, psC = np.array(sc["per_seed"][b]), np.array(sc["per_seed"][c_])
            if abs(psB.mean() - r["B"]) > 1e-6 or abs(psC.mean() - r["C"]) > 1e-6:
                stop(f"seed-mean AUROC disagrees with A_vs_C at {fam} {name} {era}: "
                     f"B {psB.mean():.6f} vs {r['B']:.6f}, C {psC.mean():.6f} vs {r['C']:.6f}")
            lo95, hi95 = r["B_minus_C_ci95"]
            verdict = "sup." if lo95 > 0 else ("NI" if r["B_minus_C_ci90lo"] > -MARGIN else "--")
            rows.append(dict(
                label=name, family=FAM[fam], era=era, events=r["events"], A=r["A"],
                B=r["B"], B_sd=psB.std(ddof=1), C=r["C"], C_sd=psC.std(ddof=1),
                gain_C=r["gain_C"], gain_C_lo=r["gain_C_ci95"][0], gain_C_hi=r["gain_C_ci95"][1],
                gain_B=r["gain_B"], gain_B_lo=r["gain_B_ci95"][0], gain_B_hi=r["gain_B_ci95"][1],
                BmC=r["B_minus_C"], BmC_lo=lo95, BmC_hi=hi95, BmC_ci90lo=r["B_minus_C_ci90lo"], verdict=verdict,
                A_passes_NI=r["A_vs_C_noninferior"]))
print(f"table 1: {len(rows)} cells; seed means match the inference files; margin {MARGIN}, "
      f"NN LR {{V: {nn_run[V]['best_lr']['B']}, M: {nn_run[M]['best_lr']['B']}}}, "
      f"tree eta {tree_run[V]['eta']['B_xgb']}")

f3 = lambda v: f"{v:.3f}"
s3 = lambda v: "0.000" if round(v, 3) == 0 else f"{v:+.3f}"
ci = lambda v, lo, hi: f"{s3(v)} [{s3(lo)}, {s3(hi)}]"
with open(OUT / "table_accuracy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
tex = [r"\begin{tabular}{lllrrrrrll}", r"\toprule",
       r"Label & Model & Era & Events & Source & B & C & C$-$Source [95\% CI] & B$-$C [95\% CI] & \\", r"\midrule"]
prev = None
for r in rows:
    key = (r["label"], r["family"])
    if prev and key != prev:
        tex.append(r"\addlinespace")
    prev = key
    tex.append(f"{r['label']} & {r['family']} & {r['era']} & {r['events']} & {f3(r['A'])} & "
               f"{f3(r['B'])}$\\pm${r['B_sd']:.3f} & {f3(r['C'])}$\\pm${r['C_sd']:.3f} & "
               f"{ci(r['gain_C'], r['gain_C_lo'], r['gain_C_hi'])} & "
               f"{ci(r['BmC'], r['BmC_lo'], r['BmC_hi'])} & {r['verdict']} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_accuracy.tex").write_text("\n".join(tex) + "\n")

# ── Table 2: pooled contrasts ───────────────────────────────────────────────
P = pooled["events"]
prow = []
for _, name, short, _ in LABELS:
    for fam in ("neural", "trees", "both"):
        d_, g_, b_ = P[f"{short}.{fam}.D"], P[f"{short}.{fam}.GC"], P[f"{short}.{fam}.GB"]
        prow.append(dict(label=name, model={"neural": "Neural", "trees": "Trees", "both": "Both"}[fam],
                         BmC=d_["est"], BmC_lo=d_["lo"], BmC_hi=d_["hi"], BmC_p=d_["p"],
                         CmA=g_["est"], CmA_lo=g_["lo"], CmA_hi=g_["hi"], CmA_p=g_["p"],
                         BmA=b_["est"], BmA_lo=b_["lo"], BmA_hi=b_["hi"], BmA_p=b_["p"]))
with open(OUT / "table_pooled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prow[0]))
    w.writeheader()
    w.writerows(prow)
tex = [r"\begin{tabular}{lllll}", r"\toprule",
       r"Label & Model & B$-$C [95\% CI] & C$-$Source [95\% CI] & B$-$Source [95\% CI] \\", r"\midrule"]
for r in prow:
    tex.append(f"{r['label']} & {r['model']} & {ci(r['BmC'], r['BmC_lo'], r['BmC_hi'])} & "
               f"{ci(r['CmA'], r['CmA_lo'], r['CmA_hi'])} & {ci(r['BmA'], r['BmA_lo'], r['BmA_hi'])} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_pooled.tex").write_text("\n".join(tex) + "\n")

# ── Table 3: drift detector ─────────────────────────────────────────────────
tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
       r"Era & C2ST all & C2ST physiology & C2ST treatment & BBSD $p$ (tree) & BBSD $p$ (NN) & Drift \\", r"\midrule"]
drow = []
for s in drift["steps"]:
    ca, cp, ct = s["c2st_all"], s["c2st_physio"], s["c2st_treat"]
    drow.append(dict(era=s["era"], all=ca["auroc"], all_lo=ca["lo95"], physio=cp["auroc"], physio_lo=cp["lo95"],
                     treat=ct["auroc"], treat_lo=ct["lo95"], bbsd_tree=s["bbsd"]["tree"]["p_min_bonf"],
                     bbsd_nn=s["bbsd"]["nn"]["p_min_bonf"], fire=s["fire"], scope=s["scope_suggested"]))
    pv = lambda p: f"{p:.2f}" if p >= 0.01 else f"{p:.0e}"
    tex.append(f"{s['era']} & {ca['auroc']:.3f} ({ca['lo95']:.3f}) & {cp['auroc']:.3f} ({cp['lo95']:.3f}) & "
               f"{ct['auroc']:.3f} ({ct['lo95']:.3f}) & {pv(s['bbsd']['tree']['p_min_bonf'])} & "
               f"{pv(s['bbsd']['nn']['p_min_bonf'])} & {'yes' if s['fire'] else 'no'} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_drift.tex").write_text("\n".join(tex) + "\n")
with open(OUT / "table_drift.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(drow[0]))
    w.writeheader()
    w.writerows(drow)

# ── Table 4: explanation stability (final era) ──────────────────────────────
EXP = {("trees", V): jl(V / "explanation_stability_eta0.03.json"), ("neural", V): jl(V / "nn_explanation_stability_lr3e-3.json"),
       ("trees", M): jl(M / "explanation_stability_eta0.03.json"), ("neural", M): jl(M / "nn_explanation_stability.json")}
erow = []
for lbl, name, short, d in LABELS:
    for fam in ("neural", "trees"):
        hit = [x for x in EXP[(fam, d)] if x["label"] == lbl and x["era"] == FINAL]
        if len(hit) != 1:
            stop(f"explanation stability row missing for {fam} {lbl}")
        x = hit[0]
        erow.append(dict(label=name, model=FAM[fam], n=x["n"],
                         physio_B=x["rho_physio"]["B"], physio_C=x["rho_physio"]["C"],
                         treat_B=x["rho_treat"]["B"], treat_C=x["rho_treat"]["C"],
                         all_B=x["rho_all"]["B"], all_C=x["rho_all"]["C"],
                         seeds_B_higher_physio=x["rho_physio"]["seeds_B_higher"],
                         seeds_B_higher_treat=x["rho_treat"]["seeds_B_higher"]))
with open(OUT / "table_explanation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(erow[0]))
    w.writeheader()
    w.writerows(erow)
tex = [r"\begin{tabular}{llrcccccc}", r"\toprule",
       r" & & & \multicolumn{2}{c}{Physiology} & \multicolumn{2}{c}{Treatment} & \multicolumn{2}{c}{All} \\",
       r"Label & Model & $n$ & B & C & B & C & B & C \\", r"\midrule"]
for r in erow:
    tex.append(f"{r['label']} & {r['model']} & {r['n']} & {r['physio_B']:.3f} & {r['physio_C']:.3f} & "
               f"{r['treat_B']:.3f} & {r['treat_C']:.3f} & {r['all_B']:.3f} & {r['all_C']:.3f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_explanation.tex").write_text("\n".join(tex) + "\n")

# ── Table 5: retrieval stability ────────────────────────────────────────────
RET = {V: jl(V / "retrieval_eta0.03_lr3e-3.json")["results"], M: jl(M / "retrieval_eta0.03_bc_states.json")["results"]}
rrow = []
for lbl, name, short, d in LABELS:
    for fam in ("neural", "trees"):
        hit = [x for x in RET[d] if x["label"] == lbl and x["family"] == ("trees" if fam == "trees" else "neural")]
        if len(hit) != 1:
            stop(f"retrieval row missing for {fam} {lbl}")
        x = hit[0]
        rrow.append(dict(label=name, model=FAM[fam], n_cases=x["n_cases"],
                         physio_B=x["physio"]["B"], physio_C=x["physio"]["C"], physio_seeds=x["physio"]["seeds_B_higher"],
                         treat_B=x["treat"]["B"], treat_C=x["treat"]["C"], treat_seeds=x["treat"]["seeds_B_higher"]))
with open(OUT / "table_retrieval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rrow[0]))
    w.writeheader()
    w.writerows(rrow)
tex = [r"\begin{tabular}{llrcccc}", r"\toprule",
       r" & & & \multicolumn{2}{c}{Physiology query} & \multicolumn{2}{c}{Treatment query} \\",
       r"Label & Model & Cases & B & C & B & C \\", r"\midrule"]
for r in rrow:
    tex.append(f"{r['label']} & {r['model']} & {r['n_cases']} & {r['physio_B']:.3f} & {r['physio_C']:.3f} & "
               f"{r['treat_B']:.3f} & {r['treat_C']:.3f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_retrieval.tex").write_text("\n".join(tex) + "\n")

# ── Figure: incident AUROC by era ───────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4), squeeze=False)
for j, (_, name, _, _) in enumerate(LABELS):
    for i, fam in enumerate(("Neural", "Trees")):
        rs = [r for r in rows if r["label"] == name and r["family"] == fam]
        ax, x = axes[i, j], np.arange(3)
        ax.plot(x, [r["A"] for r in rs], color=INK2, lw=1.2, ls="--", marker="D", ms=5, mfc="white",
                label="Source (never updated)", zorder=2)
        for key, lab, col, mk, dx in (("B", "B: bounded update", "#2a78d6", "o", -0.07),
                                      ("C", "C: full update", "#eb6834", "s", 0.07)):
            ax.errorbar(x + dx, [r[key] for r in rs], yerr=[r[f"{key}_sd"] for r in rs], color=col, lw=2,
                        marker=mk, ms=6, mec="white", mew=1.0, capsize=0, elinewidth=1.2, label=lab, zorder=3)
        ax.set_xticks(x, [f"{r['era']}\n({r['events']} ev.)" for r in rs], fontsize=7.5)
        ax.set_title(f"{fam} — {name}", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_ylabel("Incident AUROC" if i == 0 else "Incident AUROC")
axes[1, 2].set_ylabel("")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.4)
fig.savefig(OUT / "fig_auroc_matched.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_auroc_matched.png", dpi=200, bbox_inches="tight")

# ── facts for the text ──────────────────────────────────────────────────────
facts = dict(margin=MARGIN, seeds=SEEDS, nn_lr={"vaso_intub": nn_run[V]["best_lr"]["B"], "mortality": nn_run[M]["best_lr"]["B"]},
             tree_eta=tree_run[V]["eta"]["B_xgb"], n_subjects=pooled["n_subjects"],
             detector=dict(rule=drift["rule"], false_alarm_rate=drift["false_alarm_rate"],
                           null_auroc_mean=drift["null_auroc_mean"], adjacent_years_auroc=drift["adjacent_source_years_auroc"]),
             A_passes_NI_cells=int(sum(r["A_passes_NI"] for r in rows)), cells=len(rows),
             B_NI_or_sup_cells=int(sum(r["verdict"] in ("NI", "sup.") for r in rows)))
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))

# ── print ───────────────────────────────────────────────────────────────────
print(f"\n{'label':<12}{'model':<7}{'era':<12}{'ev':>4} {'A':>6} {'B':>6} {'C':>6}  C-A [95% CI]                 B-C [95% CI]                 ")
for r in rows:
    print(f"{r['label']:<12}{r['family']:<7}{r['era']:<12}{r['events']:>4} {r['A']:.3f} {r['B']:.3f} {r['C']:.3f}  "
          f"{ci(r['gain_C'], r['gain_C_lo'], r['gain_C_hi']):<28} {ci(r['BmC'], r['BmC_lo'], r['BmC_hi']):<28} {r['verdict']}"
          f"{'  A-passes-NI' if r['A_passes_NI'] else ''}")
print(f"\nB non-inferior or superior to C in {facts['B_NI_or_sup_cells']}/{facts['cells']} cells; "
      f"never-updated model passes the same test in {facts['A_passes_NI_cells']}/{facts['cells']}")
print("\npooled (event-weighted, subject-cluster bootstrap):")
for r in prow:
    print(f"  {r['label']:<11}{r['model']:<7} B-C {ci(r['BmC'], r['BmC_lo'], r['BmC_hi'])} p={r['BmC_p']:.3f} | "
          f"C-A {ci(r['CmA'], r['CmA_lo'], r['CmA_hi'])} p={r['CmA_p']:.3f}")
print("\nexplanation stability, final era (Spearman vs source; B bounded, C full):")
for r in erow:
    print(f"  {r['label']:<11}{r['model']:<7} physio {r['physio_B']:.3f}/{r['physio_C']:.3f} (B>C {r['seeds_B_higher_physio']}/5)  "
          f"treat {r['treat_B']:.3f}/{r['treat_C']:.3f} (B>C {r['seeds_B_higher_treat']}/5)  all {r['all_B']:.3f}/{r['all_C']:.3f}")
print("\nretrieval Jaccard vs source (B/C):")
for r in rrow:
    print(f"  {r['label']:<11}{r['model']:<7} n={r['n_cases']}  physio {r['physio_B']:.3f}/{r['physio_C']:.3f} (B>C {r['physio_seeds']}/5)  "
          f"treat {r['treat_B']:.3f}/{r['treat_C']:.3f} (B>C {r['treat_seeds']}/5)")
print(f"\nSaved → {OUT}/: table_accuracy, table_pooled, table_drift, table_explanation, table_retrieval (.csv/.tex), "
      f"fig_auroc_matched.pdf/.png, facts.json")
