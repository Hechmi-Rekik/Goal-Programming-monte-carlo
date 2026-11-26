"""Représentations graphiques de l'analyse de sensibilité (à exécuter après monte_carlo.py)."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

DOSSIER = ""
df = pd.read_pickle("simulations.pkl")
syn = pd.read_pickle("syntheses.pkl")

BLEU, ORANGE, VERT_EAU = "#257de0", "#f47230", "#00b894"
MARINE, GRIS, GRIS_CLAIR = "#1f3b5c", "#666666", "#e6e9ef"
COUL_TYPE = {"S1": BLEU, "S2": ORANGE, "S3": VERT_EAU}
LIB_TYPE = {"S1": "S1 : plafond Key Oil respecté",
            "S2": "S2 : rentabilité atteinte",
            "S3": "S3 : deux objectifs atteints"}
df["t"] = df["type"].str[:2]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlecolor": MARINE, "axes.edgecolor": "#b0b7c3",
    "axes.labelcolor": "#333333", "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRIS_CLAIR, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def sauver(fig, nom):
    fig.savefig(DOSSIER + nom, dpi=200, facecolor="white")
    plt.close(fig)


# ---------- Figure 1 : résolution graphique du modèle initial ----------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
x = np.linspace(0, 1000, 400)
ax.add_patch(Polygon([[0, 0], [1000, 0], [0, 500]], closed=True,
                     facecolor="#dbe8f8", edgecolor="none", alpha=0.8,
                     label="Région réalisable (capital ≤ 50 000 $)"))
ax.plot(x, (50000 - 50 * x) / 100, color=MARINE, lw=2, label="Capital : 50x₁ + 100x₂ = 50 000")
ax.plot(x, (4500 - 3 * x) / 10, color=ORANGE, lw=2, ls="--", label="But 1 : 3x₁ + 10x₂ = 4 500")
ax.axhline(300, color=VERT_EAU, lw=2, ls="-.", label="But 2 : 100x₂ = 30 000")
ax.scatter([400], [300], s=90, color=BLEU, zorder=5, edgecolor="white", linewidth=2)
ax.scatter([250], [375], s=90, color=ORANGE, zorder=5, edgecolor="white", linewidth=2)
ax.annotate("A (400 ; 300)\nsolution optimale\nZ* = 4,67 %", (400, 300), xytext=(640, 360),
            fontsize=9, color=MARINE, arrowprops=dict(arrowstyle="->", color=GRIS))
ax.annotate("B (250 ; 375)\nZ = 7,50 %", (250, 375), xytext=(320, 440),
            fontsize=9, color=MARINE, arrowprops=dict(arrowstyle="->", color=GRIS))
ax.set_xlim(0, 1000); ax.set_ylim(0, 520)
ax.set_xlabel("x₁ : actions AGA Products"); ax.set_ylabel("x₂ : actions Key Oil")
ax.set_title("Résolution graphique du GP pondéré initial")
ax.legend(loc="upper right", fontsize=8.5)
sauver(fig, "fig01_resolution_graphique.png")


# ---------- Figure 2 : schéma de la démarche Monte-Carlo ----------
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
etapes = [
    ("Tirage conjoint", "w₁ ~ U[0,50 ; 0,90]\nw₂ = 1 − w₁\nr* ~ U[8 % ; 10 %]\nk* ~ U[50 % ; 70 %]", BLEU),
    ("Construction", "Buts R = r*·50 000\nK = k*·50 000\nPoids normalisés\nw₁/R et w₂/K", "#3a8fd6"),
    ("Résolution", "Programme linéaire\n(HiGHS / SciPy)\nx₁, x₂, n₁, p₁, n₂, p₂", VERT_EAU),
    ("Enregistrement", "Colonne s de la\nmatrice des\nrésultats\n(variables ×\nscénarios)", ORANGE),
    ("Analyse", "Statistiques,\nfréquences,\natteinte des buts,\ncorrélations", "#6a4ce2"),
]
for i, (titre, texte, c) in enumerate(etapes):
    x0 = 0.1 + i * 2.0
    ax.add_patch(FancyBboxPatch((x0, 0.35), 1.7, 2.45, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=c, edgecolor="none", alpha=0.12))
    ax.add_patch(FancyBboxPatch((x0, 2.3), 1.7, 0.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=c, edgecolor="none"))
    ax.text(x0 + 0.85, 2.55, titre, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(x0 + 0.85, 1.3, texte, ha="center", va="center", color="#222222", fontsize=8.6, linespacing=1.5)
    if i < 4:
        ax.annotate("", xy=(x0 + 2.0, 1.55), xytext=(x0 + 1.75, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.6))
ax.annotate("", xy=(0.95, 0.25), xytext=(6.95, 0.25),
            arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2, connectionstyle="arc3,rad=0"))
ax.text(4.0, 0.06, "répété pour s = 1, 2, …, 10 000", ha="center", fontsize=9, color=GRIS, style="italic")
sauver(fig, "fig02_schema_monte_carlo.png")


# ---------- Figure 3 : histogrammes des variables de décision ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, v, lib, init, c in [(axes[0], "x1", "x₁ : actions AGA Products", 400, BLEU),
                            (axes[1], "x2", "x₂ : actions Key Oil", 300, ORANGE)]:
    ax.hist(df[v], bins=40, color=c, edgecolor="white", linewidth=0.8)
    ax.axvline(init, color=MARINE, lw=2, ls="--", label=f"solution initiale = {init}")
    ax.axvline(df[v].mean(), color="#ec4647", lw=1.6, label=f"moyenne = {df[v].mean():.1f}")
    ax.legend(fontsize=8.5, loc="upper left" if v == "x1" else "upper right")
    ax.set_xlabel(lib); ax.set_ylabel("Nombre de scénarios")
    ax.set_title(f"Distribution de {v} (10 000 scénarios)")
sauver(fig, "fig03_histogrammes_decisions.png")


# ---------- Figure 4 : histogrammes de Z, du rendement et de la part Key Oil ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, v, lib, init, c in [(axes[0], "Z", "Z* (%)", 4.6667, "#6a4ce2"),
                            (axes[1], "rendement_%", "Rendement annuel (% du capital)", 8.4, VERT_EAU),
                            (axes[2], "part_KeyOil_%", "Part investie dans Key Oil (%)", 60, ORANGE)]:
    ax.hist(df[v], bins=40, color=c, edgecolor="white", linewidth=0.8)
    ax.axvline(init, color=MARINE, lw=2, ls="--", label="valeur initiale")
    ax.set_xlabel(lib); ax.set_ylabel("Nombre de scénarios")
    ax.legend(fontsize=8)
axes[0].set_title("Valeur optimale Z*"); axes[1].set_title("Rendement obtenu"); axes[2].set_title("Exposition à Key Oil")
sauver(fig, "fig04_histogrammes_resultats.png")


# ---------- Figure 5 : boxplots ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2, 2, 1.2]})
def boites(ax, data, noms, couleurs):
    b = ax.boxplot(data, tick_labels=noms, patch_artist=True, widths=0.5,
                   medianprops=dict(color=MARINE, lw=2),
                   flierprops=dict(marker="o", markersize=2.5, markerfacecolor=GRIS, markeredgecolor="none", alpha=0.4))
    for p, c in zip(b["boxes"], couleurs):
        p.set_facecolor(c); p.set_alpha(0.75); p.set_edgecolor(c)
boites(axes[0], [df.x1, df.x2], ["x₁ (AGA)", "x₂ (Key Oil)"], [BLEU, ORANGE])
axes[0].set_title("Variables de décision"); axes[0].set_ylabel("Nombre d'actions")
boites(axes[1], [df.n1 / (df.r_star * 50000) * 100, df.p2 / (df.k_star * 50000) * 100],
       ["n₁ / R (%)", "p₂ / K (%)"], [VERT_EAU, "#ec4647"])
axes[1].set_title("Écarts indésirables relatifs"); axes[1].set_ylabel("% du niveau d'aspiration")
boites(axes[2], [df.Z], ["Z* (%)"], ["#6a4ce2"])
axes[2].set_title("Fonction objectif")
sauver(fig, "fig05_boxplots.png")

# boxplots de x2 par type de solution
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ordre = ["S1", "S2", "S3"]
b = ax.boxplot([df.loc[df.t == t, "x2"] for t in ordre], tick_labels=[LIB_TYPE[t] for t in ordre],
               patch_artist=True, widths=0.5, medianprops=dict(color=MARINE, lw=2),
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=GRIS, markeredgecolor="none"))
for p, t in zip(b["boxes"], ordre):
    p.set_facecolor(COUL_TYPE[t]); p.set_alpha(0.75); p.set_edgecolor(COUL_TYPE[t])
ax.axhline(300, color=MARINE, ls="--", lw=1.4)
ax.text(2.3, 303, "x₂ initial = 300", fontsize=8.5, color=MARINE)
ax.set_ylabel("x₂ : actions Key Oil"); ax.set_title("Nombre d'actions Key Oil selon le type de solution")
ax.tick_params(axis="x", labelsize=8.5)
sauver(fig, "fig06_boxplot_x2_par_type.png")


# ---------- Figure 7 : fréquences des types de solution et atteinte des objectifs ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ft = df.t.value_counts().reindex(ordre) / len(df) * 100
bars = axes[0].bar(["S1\nplafond Key Oil\nrespecté", "S2\nrentabilité\natteinte", "S3\ndeux objectifs\natteints"], ft.values,
                   color=[COUL_TYPE[t] for t in ordre], width=0.6)
for bb, v in zip(bars, ft.values):
    axes[0].text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.2f} %", ha="center", fontsize=9.5, color="#222")
axes[0].set_ylim(0, 70); axes[0].set_ylabel("% des scénarios")
axes[0].set_title("Fréquence des types de solution")
at = syn["atteinte"]["Pourcentage (%)"]
labs = ["Objectif 1\nrentabilité", "Objectif 2\nrisque", "Les deux\nobjectifs", "Aucun\nobjectif"]
bars = axes[1].bar(labs, at.values, color=[ORANGE, BLEU, VERT_EAU, "#b0b7c3"], width=0.6)
for bb, v in zip(bars, at.values):
    axes[1].text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.2f} %", ha="center", fontsize=9.5, color="#222")
axes[1].set_ylim(0, 95); axes[1].set_ylabel("% des scénarios")
axes[1].set_title("Taux de réalisation des objectifs")
sauver(fig, "fig07_frequences_atteinte.png")


# ---------- Figure 8 : cartes de décision ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ech = df.sample(4000, random_state=1)
for t in ordre:
    d = ech[ech.t == t]
    axes[0].scatter(d.k_star * 100, d.r_star * 100, s=7, color=COUL_TYPE[t], alpha=0.65, label=LIB_TYPE[t])
kk = np.linspace(50, 70, 10)
axes[0].plot(kk, 6 + 0.04 * kk, color=MARINE, lw=2)
axes[0].text(63.5, 8.06, "frontière de conflit\nr* = 6 % + 0,04·k*", fontsize=8.5, color=MARINE,
             bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.9))
axes[0].scatter([60], [9], marker="*", s=220, color="#ec4647", edgecolor="white", zorder=5, label="paramètres initiaux")
axes[0].set_xlabel("k* : plafond Key Oil (%)"); axes[0].set_ylabel("r* : rentabilité visée (%)")
axes[0].set_title("Plan des niveaux d'aspiration (k*, r*)")
h, l = axes[0].get_legend_handles_labels()
conf = ech[ech.t != "S3"]
for t in ["S1", "S2"]:
    d = conf[conf.t == t]
    axes[1].scatter(d.seuil, d.ratio_w, s=7, color=COUL_TYPE[t], alpha=0.65, label=LIB_TYPE[t])
lim = [3.0, 5.2]
axes[1].plot(lim, lim, color=MARINE, lw=2)
axes[1].text(4.62, 5.6, "w₁/w₂ = 25·r*/k*", fontsize=8.5, color=MARINE,
             bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.9))
axes[1].scatter([3.75], [0.7 / 0.3], marker="*", s=220, color="#ec4647", edgecolor="white", zorder=5)
axes[1].set_xlim(lim)
axes[1].set_xlabel("Seuil de basculement 25·r*/k*"); axes[1].set_ylabel("Rapport des poids w₁/w₂")
axes[1].set_title("Scénarios en conflit : règle de basculement")
from matplotlib.lines import Line2D
poignees = [Line2D([], [], ls="", marker="o", ms=7, color=COUL_TYPE[t]) for t in ordre] + \
           [Line2D([], [], ls="", marker="*", ms=13, color="#ec4647")]
fig.legend(poignees, [LIB_TYPE[t] for t in ordre] + ["paramètres initiaux"], loc="lower center",
           ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.07))
sauver(fig, "fig08_cartes_decision.png")


# ---------- Figure 9 : matrice de corrélation (Spearman) ----------
cs = syn["corr_s"].drop(index="w2")
fig, ax = plt.subplots(figsize=(7.4, 3.1))
im = ax.imshow(cs.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(cs.shape[1]), ["x₁", "x₂", "n₁", "p₂", "Z*", "Rendement"])
ax.set_yticks(range(cs.shape[0]), ["w₁ (= 1 − w₂)", "r*", "k*"])
for i in range(cs.shape[0]):
    for j in range(cs.shape[1]):
        v = cs.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v) > 0.55 else "#222")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035); cb.outline.set_visible(False)
ax.set_title("Corrélations de rang de Spearman : paramètres × résultats")
sauver(fig, "fig09_correlations.png")


# ---------- Figure 10 : nuages paramètres -> x2 ----------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, p, lib in [(axes[0], "w1", "w₁"), (axes[1], "r_star", "r*"), (axes[2], "k_star", "k*")]:
    for t in ordre:
        d = ech[ech.t == t]
        ax.scatter(d[p], d.x2, s=5, color=COUL_TYPE[t], alpha=0.55, label=LIB_TYPE[t])
    ax.set_xlabel(lib)
    ax.set_title(f"x₂ en fonction de {lib}  (ρ = {syn['corr_s'].loc[p, 'x2']:+.2f})")
axes[0].set_ylabel("x₂ : actions Key Oil")
axes[0].legend(fontsize=7.5, markerscale=3, loc="upper left")
sauver(fig, "fig10_nuages_x2.png")


# ---------- Figure 11 : probabilité de basculement selon w1 ----------
classes = pd.cut(df.w1, np.arange(0.50, 0.9001, 0.025))
p = df.groupby(classes, observed=True).apply(lambda d: pd.Series({
    "S1": (d.t == "S1").mean() * 100, "S2": (d.t == "S2").mean() * 100, "S3": (d.t == "S3").mean() * 100}))
centres = [c.mid for c in p.index]
fig, ax = plt.subplots(figsize=(7.8, 3.7))
for t in ordre:
    ax.plot(centres, p[t], color=COUL_TYPE[t], lw=2, marker="o", markersize=4.5, label=LIB_TYPE[t])
ax.axvline(0.7, color=MARINE, ls="--", lw=1.3)
ax.text(0.703, 63, "w₁ initial", fontsize=8.5, color=MARINE)
ax.set_xlabel("Poids de la rentabilité w₁"); ax.set_ylabel("% des scénarios de la classe")
ax.set_title("Répartition des types de solution selon w₁")
ax.legend(fontsize=8)
sauver(fig, "fig11_types_selon_w1.png")
print("Figures générées.")
