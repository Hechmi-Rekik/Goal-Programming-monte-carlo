"""
Analyse de sensibilité du GP pondéré par simulation de Monte-Carlo
10 000 scénarios : (w1, r*, k*) tirés conjointement, w2 = 1 - w1
"""

import numpy as np
import pandas as pd
from gp_modele import resoudre_gp

N = 10_000
GRAINE = 2026
SOL_INIT = (400, 300)          # solution optimale initiale (x1, x2)


def tirer_parametres(n=N, graine=GRAINE):
    """Tirage conjoint des quatre paramètres pour chaque scénario."""
    rng = np.random.default_rng(graine)
    w1 = rng.uniform(0.50, 0.90, n)
    r = rng.uniform(0.08, 0.10, n)
    k = rng.uniform(0.50, 0.70, n)
    return w1, 1 - w1, r, k


def type_solution(row):
    if row["obj1_atteint"] and row["obj2_atteint"]:
        return "S3 : deux objectifs atteints"
    if row["obj2_atteint"]:
        return "S1 : plafond Key Oil respecté, rentabilité insuffisante"
    if row["obj1_atteint"]:
        return "S2 : rentabilité atteinte, plafond Key Oil dépassé"
    return "S0 : aucun objectif atteint"


def simuler(normaliser=True):
    w1, w2, r, k = tirer_parametres()
    lignes = []
    for s in range(N):
        sol = resoudre_gp(w1[s], r[s], k[s], normaliser=normaliser, entier=False)
        sol["scenario"] = s + 1
        lignes.append(sol)
    df = pd.DataFrame(lignes).set_index("scenario")
    df["type"] = df.apply(type_solution, axis=1)
    df["meme_structure"] = df["type"].str[:2].isin(["S1", "S3"]).astype(int)
    # solution initiale « retrouvée » : écart d'au plus une action sur x1 et x2
    df["solution_initiale"] = ((abs(df.x1 - SOL_INIT[0]) <= 1) & (abs(df.x2 - SOL_INIT[1]) <= 1)).astype(int)
    df["proche_initiale_10%"] = ((abs(df.x1 - SOL_INIT[0]) <= 0.10 * SOL_INIT[0]) &
                                 (abs(df.x2 - SOL_INIT[1]) <= 0.10 * SOL_INIT[1])).astype(int)
    # solution « proche » : écart d'au plus 5 % sur chaque variable de décision
    df["proche_initiale_5%"] = ((abs(df.x1 - SOL_INIT[0]) <= 0.05 * SOL_INIT[0]) &
                                (abs(df.x2 - SOL_INIT[1]) <= 0.05 * SOL_INIT[1])).astype(int)
    # indicateurs théoriques (cf. rapport, section 3)
    df["conflit"] = (df.r_star > 0.06 + 0.04 * df.k_star).astype(int)
    df["ratio_w"] = df.w1 / df.w2
    df["seuil"] = 25 * df.r_star / df.k_star
    return df


def statistiques(df, colonnes):
    d = df[colonnes]
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    st = pd.DataFrame({
        "Moyenne": d.mean(),
        "Écart-type": d.std(ddof=1),
        "CV (%)": d.std(ddof=1) / d.mean().abs() * 100,
        "Minimum": d.min(),
        "Q1": q1,
        "Médiane": d.median(),
        "Q3": q3,
        "Maximum": d.max(),
        "IQR": q3 - q1,
    })
    st.loc[d.mean().abs() < 1e-9, "CV (%)"] = np.nan
    return st


if __name__ == "__main__":
    df = simuler(normaliser=True)
    df.to_pickle("simulations.pkl")

    cols_stats = ["w1", "w2", "r_star", "k_star", "x1", "x2", "n1", "p1", "n2", "p2",
                  "Z", "rendement_%", "part_KeyOil_%"]
    st = statistiques(df, cols_stats)

    freq_types = df["type"].value_counts().rename("Effectif").to_frame()
    freq_types["Fréquence (%)"] = freq_types["Effectif"] / N * 100

    freq_sol = (df.assign(x1=df.x1.round(0), x2=df.x2.round(0))
                .groupby(["x1", "x2"]).size().rename("Effectif")
                .sort_values(ascending=False).reset_index())
    freq_sol["Fréquence (%)"] = freq_sol["Effectif"] / N * 100

    atteinte = pd.DataFrame({
        "Effectif": [df.obj1_atteint.sum(), df.obj2_atteint.sum(),
                     (df.obj1_atteint & df.obj2_atteint).sum(),
                     ((1 - df.obj1_atteint) & (1 - df.obj2_atteint)).sum()]},
        index=["Objectif 1 (rentabilité)", "Objectif 2 (risque)",
               "Les deux objectifs", "Aucun objectif"])
    atteinte["Pourcentage (%)"] = atteinte["Effectif"] / N * 100

    params = ["w1", "w2", "r_star", "k_star"]
    sorties = ["x1", "x2", "n1", "p2", "Z", "rendement_%"]
    corr_p = df[params + sorties].corr("pearson").loc[params, sorties]
    corr_s = df[params + sorties].corr("spearman").loc[params, sorties]

    # Variante non normalisée (comparaison)
    df_brut = simuler(normaliser=False)
    freq_brut = df_brut["type"].value_counts().rename("Effectif").to_frame()
    freq_brut["Fréquence (%)"] = freq_brut["Effectif"] / N * 100

    pd.to_pickle({"stats": st, "freq_types": freq_types, "freq_sol": freq_sol,
                  "atteinte": atteinte, "corr_p": corr_p, "corr_s": corr_s,
                  "freq_brut": freq_brut}, "syntheses.pkl")

    # ---------- Affichage console ----------
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\nStatistiques descriptives\n", st.round(4))
    print("\nTypes de solutions\n", freq_types)
    print("\n10 solutions les plus fréquentes\n", freq_sol.head(10))
    print("\nNombre de solutions distinctes :", len(freq_sol))
    print("Solution initiale exacte (400, 300) :", df.solution_initiale.sum(),
          f"({df.solution_initiale.mean():.2%})")
    print("Solution à ±5 % de l'initiale      :", df['proche_initiale_5%'].sum(),
          f"({df['proche_initiale_5%'].mean():.2%})")
    print("Solution à ±10 % de l'initiale     :", df['proche_initiale_10%'].sum())
    print("Même structure que l'initiale (S1/S3):", df.meme_structure.sum())
    print("\nAtteinte des objectifs\n", atteinte)
    print("\nCorrélations de Pearson\n", corr_p.round(3))
    print("\nCorrélations de Spearman\n", corr_s.round(3))
    print("\nScénarios en conflit :", df.conflit.sum())
    sub = df[df.conflit == 1]
    pred = np.where(sub.ratio_w > sub.seuil, "S2", "S1")
    obs = sub["type"].str[:2]
    print("Concordance règle théorique / simulation :", (pred == obs).mean())
    print("\nVariante non normalisée\n", freq_brut)
