"""
Module IA — Classification automatique des transactions comptables
TF-IDF + Random Forest — Normes OHADA
Hackathon ISM 2026 — Groupe 17
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions_pme.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models")


# ─────────────────────────────────────────────
#  1. CLASSIFICATION DES CATEGORIES COMPTABLES
# ─────────────────────────────────────────────

def entrainer_classifieur(data_path=DATA_PATH):
    """
    Entraîne un pipeline TF-IDF + Random Forest pour classifier
    automatiquement les libellés de transactions en catégories OHADA.
    Retourne le pipeline entraîné et le rapport de performances.
    """
    df = pd.read_csv(data_path)
    df = df[df["anomalie"] == 0].copy()  # on n'entraîne pas sur les anomalies

    X = df["libelle"]
    y = df["categorie"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rapport = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    # Sauvegarde
    os.makedirs(MODELS_PATH, exist_ok=True)
    with open(os.path.join(MODELS_PATH, "classifieur_categories.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline, accuracy, rapport


def charger_classifieur():
    """Charge le classifieur depuis le disque."""
    chemin = os.path.join(MODELS_PATH, "classifieur_categories.pkl")
    if not os.path.exists(chemin):
        raise FileNotFoundError("Modele non trouve. Lancez d'abord entrainer_classifieur().")
    with open(chemin, "rb") as f:
        return pickle.load(f)


def classifier_transaction(libelle: str, pipeline=None):
    """
    Prédit la catégorie OHADA d'un libellé de transaction.
    Retourne (categorie_predite, confiance, top3_categories)
    """
    if pipeline is None:
        pipeline = charger_classifieur()

    proba = pipeline.predict_proba([libelle])[0]
    classes = pipeline.classes_
    idx_max = np.argmax(proba)

    # Top 3
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(classes[i], round(proba[i] * 100, 1)) for i in top3_idx]

    return classes[idx_max], round(proba[idx_max] * 100, 1), top3


# ─────────────────────────────────────────────
#  2. DETECTION D'ANOMALIES (Isolation Forest)
# ─────────────────────────────────────────────

def entrainer_isolation_forest(data_path=DATA_PATH):
    """
    Entraîne un Isolation Forest sur les transactions normales pour
    détecter les montants inhabituels.
    """
    df = pd.read_csv(data_path)

    # Features numériques simples
    df["date"] = pd.to_datetime(df["date"])
    df["mois"]         = df["date"].dt.month
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["est_charge"]   = (df["type"] == "charge").astype(int)

    features = df[["montant", "mois", "jour_semaine", "est_charge"]]

    # Entraîner sur les transactions normales uniquement
    normales = features[df["anomalie"] == 0]

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
    )
    iso.fit(normales)

    os.makedirs(MODELS_PATH, exist_ok=True)
    with open(os.path.join(MODELS_PATH, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)

    # Score sur tout le dataset
    scores = iso.decision_function(features)
    predictions = iso.predict(features)

    df["score_anomalie"] = scores
    df["est_anomalie"]   = (predictions == -1).astype(int)

    return iso, df[["id", "date", "libelle", "montant", "score_anomalie", "est_anomalie"]]


def charger_isolation_forest():
    chemin = os.path.join(MODELS_PATH, "isolation_forest.pkl")
    if not os.path.exists(chemin):
        raise FileNotFoundError("Modele Isolation Forest non trouve.")
    with open(chemin, "rb") as f:
        return pickle.load(f)


def detecter_anomalie(montant: float, mois: int, jour_semaine: int, est_charge: bool, iso=None):
    """
    Prédit si une transaction est anormale.
    Retourne (est_anomalie: bool, score: float)
    Score < 0 = suspect, plus négatif = plus suspect.
    """
    if iso is None:
        iso = charger_isolation_forest()

    features = [[montant, mois, jour_semaine, int(est_charge)]]
    score = iso.decision_function(features)[0]
    pred  = iso.predict(features)[0]
    return pred == -1, round(score, 4)


# ─────────────────────────────────────────────
#  SCRIPT PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Entrainement du classifieur TF-IDF + Random Forest ===")
    pipeline, accuracy, rapport = entrainer_classifieur()
    print(f"Precision globale : {accuracy * 100:.1f}%")
    print("Rapport par categorie :")
    for cat, metrics in rapport.items():
        if isinstance(metrics, dict):
            print(f"  {cat[:40]:<40} F1={metrics.get('f1-score', 0):.2f}")

    print("\n=== Test de classification ===")
    tests = [
        "LOYER BOUTIQUE PLATEAU DAKAR",
        "SALAIRES EMPLOYES DU MOIS",
        "RECETTE JOURNALIERE CAISSE",
        "FACTURE SENELEC ELECTRICITE",
        "ACHAT MARCHANDISES MARCHE SANDAGA",
        "VIREMENT COMPTE TIERS INCONNU",
    ]
    for t in tests:
        cat, conf, top3 = classifier_transaction(t, pipeline)
        print(f"  '{t}'")
        print(f"    => {cat} ({conf}%)")

    print("\n=== Entrainement Isolation Forest ===")
    iso, resultats = entrainer_isolation_forest()
    anomalies_detectees = resultats[resultats["est_anomalie"] == 1]
    print(f"Transactions suspectes detectees : {len(anomalies_detectees)}")
    print(anomalies_detectees[["date", "libelle", "montant", "score_anomalie"]].to_string(index=False))
