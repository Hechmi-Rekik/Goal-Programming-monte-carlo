"""
Programmation par objectifs pondérée (Weighted Goal Programming)
Gestion de portefeuille - DJS Investment Services

Variables de décision
    x1 : nombre d'actions AGA Products achetées (prix 50 $, rentabilité 6 %)
    x2 : nombre d'actions Key Oil achetées      (prix 100 $, rentabilité 10 %)
Variables d'écart
    n1, p1 : écarts négatif / positif par rapport au but de rentabilité
    n2, p2 : écarts négatif / positif par rapport au plafond Key Oil

Modèle (déviations normalisées en pourcentage du but) :
    min Z = 100 x [ w1 * n1 / R  +  w2 * p2 / K ]   (en %)
    s.c.  50 x1 + 100 x2                 <= 50 000      (capital disponible)
          3 x1 +  10 x2 + n1 - p1        =  R            (but 1 : R = r* x 50 000)
                 100 x2 + n2 - p2        =  K            (but 2 : K = k* x 50 000)
          x1, x2, n1, p1, n2, p2 >= 0  (option entier=True : x1, x2 entiers)
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

CAPITAL = 50_000          # capital disponible ($)
PRIX = (50, 100)          # prix unitaires AGA, Key Oil ($)
RENTA = (0.06, 0.10)      # rentabilités annuelles estimées
EPS = 1e-6                # règle de départage (voir rapport, section 2.3)

# Ordre des variables : x1, x2, n1, p1, n2, p2
NOMS = ["x1", "x2", "n1", "p1", "n2", "p2"]


def resoudre_gp(w1=0.7, r_star=0.09, k_star=0.60, normaliser=True, entier=False):
    """Construit et résout le GP pondéré pour un jeu (w1, r*, k*).

    Retourne un dictionnaire contenant les variables, la valeur de Z
    et quelques indicateurs dérivés.
    """
    w2 = 1.0 - w1
    R = r_star * CAPITAL                       # but de rentabilité ($)
    K = k_star * CAPITAL                       # plafond Key Oil ($)
    g1 = PRIX[0] * RENTA[0]                    # 3 $ de gain par action AGA
    g2 = PRIX[1] * RENTA[1]                    # 10 $ de gain par action Key Oil

    # Coefficients de normalisation (déviations en % du but) ou bruts ($)
    # (facteur 100 : Z exprimé en points de pourcentage, ce qui améliore aussi
    #  le conditionnement numérique du programme linéaire)
    c1 = 100 * w1 / R if normaliser else w1
    c2 = 100 * w2 / K if normaliser else w2

    # Fonction objectif : seules les déviations indésirables n1 et p2 sont pénalisées.
    # Le terme -EPS * rentabilité sert uniquement à départager des optima multiples.
    c = np.array([-EPS * g1 / R, -EPS * g2 / R, c1, 0.0, 0.0, c2])
    c[:2] *= 100

    A = np.array([
        [PRIX[0], PRIX[1], 0, 0, 0, 0],        # contrainte de capital
        [g1, g2, 1, -1, 0, 0],                 # but 1 : rentabilité
        [0, PRIX[1], 0, 0, 1, -1],             # but 2 : plafond Key Oil
    ], dtype=float)
    lb = np.array([-np.inf, R, K])
    ub = np.array([CAPITAL, R, K])

    integrality = np.array([1, 1, 0, 0, 0, 0]) if entier else np.zeros(6)
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=integrality, bounds=Bounds(0, np.inf),
               options={"mip_rel_gap": 0, "presolve": True})
    if not res.success:
        raise RuntimeError(res.message)

    x1, x2, n1, p1, n2, p2 = np.round(res.x, 6)
    investi = PRIX[0] * x1 + PRIX[1] * x2
    rendement = g1 * x1 + g2 * x2
    Z = c1 * n1 + c2 * p2                      # valeur de Z sans le terme de départage
    tol = 1e-6
    return {
        "w1": w1, "w2": w2, "r_star": r_star, "k_star": k_star,
        "x1": x1, "x2": x2, "n1": n1, "p1": p1, "n2": n2, "p2": p2,
        "Z": Z,
        "montant_AGA": PRIX[0] * x1, "montant_KeyOil": PRIX[1] * x2,
        "capital_investi": investi,
        "rendement_$": rendement,
        "rendement_%": rendement / CAPITAL * 100,
        "part_KeyOil_%": PRIX[1] * x2 / CAPITAL * 100,
        "obj1_atteint": int(n1 <= tol),
        "obj2_atteint": int(p2 <= tol),
    }


def afficher(sol, titre):
    print("=" * 62)
    print(titre)
    print("=" * 62)
    print(f"Paramètres : w1 = {sol['w1']:.2f} | w2 = {sol['w2']:.2f} | "
          f"r* = {sol['r_star']:.0%} | k* = {sol['k_star']:.0%}")
    print("-" * 62)
    print(f"x1 (AGA Products)  = {sol['x1']:>8.0f} actions  -> {sol['montant_AGA']:>9,.0f} $")
    print(f"x2 (Key Oil)       = {sol['x2']:>8.0f} actions  -> {sol['montant_KeyOil']:>9,.0f} $")
    print(f"Capital investi    = {sol['capital_investi']:>9,.0f} $")
    print(f"Rendement annuel   = {sol['rendement_$']:>9,.0f} $  ({sol['rendement_%']:.2f} %)")
    print(f"Part Key Oil       = {sol['part_KeyOil_%']:.2f} % du capital")
    print("-" * 62)
    for v in ["n1", "p1", "n2", "p2"]:
        print(f"{v:<4} = {sol[v]:>10,.2f}")
    print(f"Z*   = {sol['Z']:.4f}")
    print(f"Objectif 1 (rentabilité) atteint : {'oui' if sol['obj1_atteint'] else 'non'}")
    print(f"Objectif 2 (risque) atteint      : {'oui' if sol['obj2_atteint'] else 'non'}")


if __name__ == "__main__":
    sol = resoudre_gp(0.7, 0.09, 0.60, normaliser=True)
    afficher(sol, "Modèle GP pondéré initial (déviations normalisées)")
    sol_brut = resoudre_gp(0.7, 0.09, 0.60, normaliser=False)
    afficher(sol_brut, "Variante : déviations non normalisées (en $)")
