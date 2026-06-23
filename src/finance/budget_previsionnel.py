"""
Module Finance — Prévision budgétaire avec Prophet (Meta)
Analyse tendance + saisonnalité des flux financiers
Hackathon ISM 2026 — Groupe 17
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions_pme.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def charger_et_preparer(data_path=DATA_PATH):
    """
    Charge le dataset et calcule les agrégats journaliers :
    - total_charges, total_produits, solde_net par jour
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Séparer charges et produits
    charges  = df[df["type"] == "charge" ].groupby("date")["montant"].sum().rename("charges")
    produits = df[df["type"] == "produit"].groupby("date")["montant"].sum().rename("produits")

    daily = pd.concat([charges, produits], axis=1).fillna(0)
    daily["solde_net"] = daily["produits"] - daily["charges"]
    daily = daily.reset_index().rename(columns={"date": "ds"})

    return daily


def entrainer_prophet(serie: pd.DataFrame, colonne: str, periodes: int = 90):
    """
    Entraîne Prophet sur une série temporelle et retourne les prévisions.

    Args:
        serie     : DataFrame avec colonnes 'ds' et la colonne cible
        colonne   : 'produits', 'charges' ou 'solde_net'
        periodes  : nombre de jours à prévoir

    Returns:
        (modele, forecast_df)
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet n'est pas installe. Lancez: pip install prophet")

    df_prophet = serie[["ds", colonne]].rename(columns={colonne: "y"})

    modele = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.90,
    )

    # Ajouter les événements saisonniers sénégalais
    modele.add_seasonality(name="saisonnalite_senegale", period=365.25, fourier_order=5)

    modele.fit(df_prophet)

    futur = modele.make_future_dataframe(periods=periodes, freq="D")
    forecast = modele.predict(futur)

    return modele, forecast


def obtenir_previsions(data_path=DATA_PATH, horizon_jours=90):
    """
    Entraîne les trois modèles (charges, produits, solde) et retourne
    un dictionnaire de résultats.
    """
    daily = charger_et_preparer(data_path)

    resultats = {}
    for col in ["produits", "charges", "solde_net"]:
        modele, forecast = entrainer_prophet(daily, col, horizon_jours)
        resultats[col] = {
            "modele":   modele,
            "forecast": forecast,
            "historique": daily[["ds", col]].rename(columns={col: "reel"}),
        }

    return resultats


def resume_previsions(resultats: dict, horizon_jours: int = 30):
    """
    Retourne un résumé des prévisions pour les N prochains jours
    après la dernière date du dataset.
    """
    # Utiliser la dernière date du dataset (pas aujourd'hui)
    derniere_date = max(
        data["historique"]["ds"].max() for data in resultats.values()
    )
    aujourd_hui = pd.Timestamp(derniere_date)
    horizon     = aujourd_hui + pd.Timedelta(days=horizon_jours)

    resume = {}
    for col, data in resultats.items():
        forecast = data["forecast"]
        futur = forecast[forecast["ds"] > aujourd_hui][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        futur = futur[futur["ds"] <= horizon]

        resume[col] = {
            "total_prevu":  int(futur["yhat"].sum()),
            "intervalle_bas": int(futur["yhat_lower"].sum()),
            "intervalle_haut": int(futur["yhat_upper"].sum()),
            "detail": futur,
        }

    return resume


def calculer_indicateurs(data_path=DATA_PATH):
    """
    Calcule les indicateurs financiers clés OHADA :
    - Chiffre d'affaires mensuel
    - Résultat net mensuel
    - Taux de marge
    - Tendance (croissance/déclin)
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["mois"] = df["date"].dt.to_period("M")

    mensuel = df.groupby(["mois", "type"])["montant"].sum().unstack(fill_value=0)
    if "produit" not in mensuel.columns:
        mensuel["produit"] = 0
    if "charge" not in mensuel.columns:
        mensuel["charge"] = 0

    mensuel["resultat_net"] = mensuel["produit"] - mensuel["charge"]
    mensuel["taux_marge"]   = (mensuel["resultat_net"] / mensuel["produit"].replace(0, np.nan) * 100).round(1)

    # Tendance sur les 3 derniers mois
    derniers = mensuel["resultat_net"].tail(3).values
    if len(derniers) >= 2:
        tendance = "hausse" if derniers[-1] > derniers[0] else "baisse"
    else:
        tendance = "stable"

    return mensuel, tendance


def sauvegarder_modeles(resultats: dict):
    os.makedirs(MODELS_PATH, exist_ok=True)
    for col, data in resultats.items():
        chemin = os.path.join(MODELS_PATH, f"prophet_{col}.pkl")
        with open(chemin, "wb") as f:
            pickle.dump(data["modele"], f)


# ─────────────────────────────────────────────
#  SCRIPT PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Prevision budgetaire — Prophet ===")
    print("Chargement et preparation des donnees...")
    daily = charger_et_preparer()
    print(f"Donnees : {len(daily)} jours de transactions")
    print(f"Charges totales  : {daily['charges'].sum():,.0f} FCFA")
    print(f"Produits totaux  : {daily['produits'].sum():,.0f} FCFA")
    print(f"Solde net total  : {daily['solde_net'].sum():,.0f} FCFA")

    print("\nEntrainement des modeles Prophet...")
    resultats = obtenir_previsions(horizon_jours=90)
    sauvegarder_modeles(resultats)
    print("Modeles sauvegardes.")

    print("\n=== Previsions — 30 prochains jours ===")
    resume = resume_previsions(resultats, horizon_jours=30)
    for col, data in resume.items():
        print(f"\n  {col.upper()}")
        print(f"    Prevu    : {data['total_prevu']:>12,.0f} FCFA")
        print(f"    Min      : {data['intervalle_bas']:>12,.0f} FCFA")
        print(f"    Max      : {data['intervalle_haut']:>12,.0f} FCFA")

    print("\n=== Indicateurs mensuels ===")
    mensuel, tendance = calculer_indicateurs()
    print(mensuel[["produit", "charge", "resultat_net", "taux_marge"]].tail(6).to_string())
    print(f"\nTendance des 3 derniers mois : {tendance.upper()}")
