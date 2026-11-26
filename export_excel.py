"""Export de la matrice des résultats (variables × scénarios) et des indicateurs vers Excel."""

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

df = pd.read_pickle("simulations.pkl")
syn = pd.read_pickle("syntheses.pkl")
FICHIER = "resultats_monte_carlo.xlsx"

lignes = {"w1": "Poids objectif 1 (w1)", "w2": "Poids objectif 2 (w2)",
          "r_star": "Rentabilité visée r*", "k_star": "Plafond Key Oil k*",
          "x1": "x1 : actions AGA", "x2": "x2 : actions Key Oil",
          "n1": "n1", "p1": "p1", "n2": "n2", "p2": "p2", "Z": "Z* (%)",
          "rendement_$": "Rendement ($)", "rendement_%": "Rendement (%)",
          "part_KeyOil_%": "Part Key Oil (%)", "obj1_atteint": "Objectif 1 atteint (1/0)",
          "obj2_atteint": "Objectif 2 atteint (1/0)", "type": "Type de solution"}
matrice = df[list(lignes)].rename(columns=lignes).T
matrice.columns = [f"S{s}" for s in matrice.columns]

par_type = df.groupby(df["type"].str[:2])[["w1", "r_star", "k_star", "x1", "x2", "n1", "p2", "Z",
                                           "rendement_%", "part_KeyOil_%"]].mean()
par_type.insert(0, "Effectif", df["type"].str[:2].value_counts())

with pd.ExcelWriter(FICHIER, engine="openpyxl") as w:
    pd.DataFrame({
        "Élément": ["Nombre de scénarios", "Graine aléatoire", "w1", "w2", "r*", "k*",
                    "Solveur", "Normalisation"],
        "Valeur": [len(df), 2026, "Uniforme [0,50 ; 0,90]", "1 − w1", "Uniforme [8 % ; 10 %]",
                   "Uniforme [50 % ; 70 %]", "HiGHS (scipy.optimize.milp)",
                   "Déviations en % du niveau d'aspiration"]}).to_excel(w, "Paramètres", index=False)
    matrice.to_excel(w, "Matrice", index_label="Variable / Scénario")
    syn["stats"].to_excel(w, "Statistiques", index_label="Variable")
    syn["freq_types"].to_excel(w, "Types de solution", index_label="Type")
    par_type.to_excel(w, "Moyennes par type", index_label="Type")
    syn["freq_sol"].head(50).to_excel(w, "Solutions fréquentes", index=False)
    syn["atteinte"].to_excel(w, "Atteinte objectifs", index_label="Objectif")
    syn["corr_p"].to_excel(w, "Corrélations", index_label="Pearson", startrow=0)
    syn["corr_s"].to_excel(w, "Corrélations", index_label="Spearman", startrow=7)

wb = load_workbook(FICHIER)
entete = PatternFill("solid", fgColor="1F3B5C")
bande = PatternFill("solid", fgColor="EAF2FB")
fin = Side(style="thin", color="C9D3E0")
for ws in wb.worksheets:
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = entete
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    if ws.title == "Corrélations":
        for c in ws[8]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = entete
    lim_col = min(ws.max_column, 60)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=lim_col):
        for c in row:
            c.border = Border(top=fin, bottom=fin, left=fin, right=fin)
            if isinstance(c.value, float):
                c.number_format = "0.0000"
        if row[0].row % 2 == 0 and ws.title != "Matrice":
            for c in row:
                c.fill = bande
        row[0].font = Font(bold=True, color="1F3B5C")
    ws.column_dimensions["A"].width = 30
    for j in range(2, lim_col + 1):
        ws.column_dimensions[get_column_letter(j)].width = 14
    ws.freeze_panes = "B2"
ws = wb["Matrice"]
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for c in row[1:]:
        if isinstance(c.value, float):
            c.number_format = "0.0000"
wb.save(FICHIER)
print("Excel écrit :", FICHIER, matrice.shape)
print(par_type.round(3).to_string())
