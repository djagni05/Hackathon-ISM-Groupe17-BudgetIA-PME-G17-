"""
Générateur de dataset simulé — PME sénégalaise (Commerce/Restauration)
Normes comptables OHADA — Hackathon ISM 2026 — Groupe 17
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# === Période ===
DEBUT = datetime(2025, 1, 1)
FIN   = datetime(2025, 12, 31)

# === Catégories OHADA (plan comptable SYSCOHADA) ===
CATEGORIES = {
    # --- CHARGES (Classe 6) ---
    "601 - Achats marchandises":      {"type": "charge", "moy": 280000, "std": 80000},
    "621 - Salaires et charges":      {"type": "charge", "moy": 300000, "std": 20000},
    "624 - Transport":                {"type": "charge", "moy": 28000,  "std": 10000},
    "626 - Télécommunications":       {"type": "charge", "moy": 25000,  "std": 5000},
    "627 - Publicité et promotion":   {"type": "charge", "moy": 45000,  "std": 25000},
    "631 - Loyer local commercial":   {"type": "charge", "moy": 150000, "std": 0},
    "641 - Eau et électricité":       {"type": "charge", "moy": 40000,  "std": 12000},
    "658 - Charges diverses":         {"type": "charge", "moy": 15000,  "std": 8000},
    # --- PRODUITS (Classe 7) ---
    "701 - Ventes de marchandises":   {"type": "produit", "moy": 180000, "std": 60000},
    "706 - Prestations de services":  {"type": "produit", "moy": 85000,  "std": 35000},
}

# === Événements saisonniers sénégalais 2025 ===
EVENEMENTS = [
    (datetime(2025, 3,  1),  datetime(2025, 3, 29),  1.40, "Ramadan"),
    (datetime(2025, 3, 30),  datetime(2025, 4,  2),  1.90, "Korité"),
    (datetime(2025, 6,  6),  datetime(2025, 6,  9),  2.10, "Tabaski"),
    (datetime(2025, 7,  4),  datetime(2025, 7,  6),  1.20, "Tamkharit"),
    (datetime(2025, 9,  1),  datetime(2025, 9, 15),  1.35, "Rentrée scolaire"),
    (datetime(2025, 12, 15), datetime(2025, 12, 31), 1.55, "Fêtes fin d'année"),
]

LIBELLES = {
    "601 - Achats marchandises": [
        "ACHAT MARCHANDISES MARCHE SANDAGA",
        "FOURNISSEUR DIALLO IMPORT-EXPORT",
        "APPROVISIONNEMENT STOCKS GROSSISTE",
        "ACHAT PRODUITS ALIMENTAIRES HLM",
        "FOURNISSEUR BA ET FRERES",
    ],
    "621 - Salaires et charges": [
        "SALAIRES EMPLOYES DU MOIS",
        "PAIEMENT SALAIRES PERSONNEL",
        "REMUNERATION MENSUELLE STAFF",
        "SALAIRES + CHARGES SOCIALES IPRES",
    ],
    "624 - Transport": [
        "TRANSPORT MARCHANDISES",
        "FRAIS LIVRAISON CLIENT DAKAR",
        "CARBURANT VEHICULE UTILITAIRE",
        "COURSE TAXI PROFESSIONNEL",
        "FRAIS PORT CONTENEUR",
    ],
    "626 - Télécommunications": [
        "FACTURE SONATEL ORANGE PRO",
        "ABONNEMENT INTERNET EXPRESSO",
        "FACTURE MOBILE MONEY WAVE",
        "FORFAIT TIGO BUSINESS",
    ],
    "627 - Publicité et promotion": [
        "IMPRESSION FLYERS ET AFFICHES",
        "PUBLICITE FACEBOOK ADS",
        "PANNEAU PUBLICITAIRE ROUTE",
        "PROMOTION SPECIALE CLIENTS",
        "SPONSORING EVENEMENT LOCAL",
    ],
    "631 - Loyer local commercial": [
        "LOYER BOUTIQUE PLATEAU DAKAR",
        "LOYER MENSUEL LOCAL COMMERCIAL",
        "LOYER MAGASIN GRAND YOFF",
    ],
    "641 - Eau et électricité": [
        "FACTURE SENELEC ELECTRICITE",
        "FACTURE SDE EAU POTABLE",
        "ELECTRICITE ET EAU MENSUELLE",
    ],
    "658 - Charges diverses": [
        "FOURNITURES ET MATERIEL BUREAU",
        "FRAIS BANCAIRES CBAO",
        "ENTRETIEN ET REPARATION MATERIEL",
        "DIVERS ET IMPREVU",
        "FRAIS NOTAIRE ET JURIDIQUE",
    ],
    "701 - Ventes de marchandises": [
        "RECETTE JOURNALIERE CAISSE",
        "VENTES MARCHANDISES COMPTOIR",
        "ENCAISSEMENT CLIENT YOFF",
        "VENTE AU DETAIL BOUTIQUE",
        "RECETTE WAVE ORANGE MONEY",
    ],
    "706 - Prestations de services": [
        "PRESTATION CLIENT ENTREPRISE",
        "HONORAIRES CONSEIL ET FORMATION",
        "SERVICE LIVRAISON A DOMICILE",
        "LOCATION MATERIEL",
    ],
}

def get_evenement(date):
    for debut, fin, mult, nom in EVENEMENTS:
        if debut <= date <= fin:
            return mult, nom
    return 1.0, None

def montant_aleatoire(cat, multiplicateur=1.0):
    moy = CATEGORIES[cat]["moy"] * multiplicateur
    std = CATEGORIES[cat]["std"]
    if std == 0:
        return int(moy)
    valeur = int(np.random.normal(moy, std))
    return max(valeur, 1000)

def libelle(cat, evenement=None):
    base = random.choice(LIBELLES[cat])
    if evenement:
        base += f" ({evenement.upper()})"
    return base

def generer_transactions():
    rows = []
    date = DEBUT

    while date <= FIN:
        mult, evt = get_evenement(date)
        jour_semaine = date.weekday()  # 0=lundi, 6=dimanche

        # ── Charges fixes le 1er de chaque mois ──────────────────────────
        if date.day == 1:
            for cat in ["631 - Loyer local commercial",
                        "621 - Salaires et charges",
                        "626 - Télécommunications",
                        "641 - Eau et électricité"]:
                rows.append({
                    "date":      date.strftime("%Y-%m-%d"),
                    "libelle":   libelle(cat),
                    "categorie": cat,
                    "type":      "charge",
                    "montant":   montant_aleatoire(cat),
                    "anomalie":  0,
                })

        # ── Achats marchandises (lun / mer / ven) ────────────────────────
        if jour_semaine in [0, 2, 4]:
            rows.append({
                "date":      date.strftime("%Y-%m-%d"),
                "libelle":   libelle("601 - Achats marchandises", evt),
                "categorie": "601 - Achats marchandises",
                "type":      "charge",
                "montant":   montant_aleatoire("601 - Achats marchandises", 1.2 if evt else 1.0),
                "anomalie":  0,
            })

        # ── Ventes quotidiennes (sauf dimanche) ──────────────────────────
        if jour_semaine != 6:
            rows.append({
                "date":      date.strftime("%Y-%m-%d"),
                "libelle":   libelle("701 - Ventes de marchandises", evt),
                "categorie": "701 - Ventes de marchandises",
                "type":      "produit",
                "montant":   montant_aleatoire("701 - Ventes de marchandises", mult),
                "anomalie":  0,
            })

        # ── Charges variables aléatoires (~20% des jours) ─────────────────
        if random.random() < 0.20:
            cat = random.choice(["624 - Transport",
                                 "627 - Publicité et promotion",
                                 "658 - Charges diverses"])
            rows.append({
                "date":      date.strftime("%Y-%m-%d"),
                "libelle":   libelle(cat),
                "categorie": cat,
                "type":      "charge",
                "montant":   montant_aleatoire(cat),
                "anomalie":  0,
            })

        # ── Prestations de services (~7% des jours) ───────────────────────
        if random.random() < 0.07:
            rows.append({
                "date":      date.strftime("%Y-%m-%d"),
                "libelle":   libelle("706 - Prestations de services"),
                "categorie": "706 - Prestations de services",
                "type":      "produit",
                "montant":   montant_aleatoire("706 - Prestations de services"),
                "anomalie":  0,
            })

        date += timedelta(days=1)

    # ── Anomalies (pour Isolation Forest) ────────────────────────────────
    anomalies = [
        {
            "date": "2025-02-14", "libelle": "VENTE NUIT HEURE INHABITUELLE",
            "categorie": "701 - Ventes de marchandises", "type": "produit",
            "montant": 1_350_000, "anomalie": 1,
        },
        {
            "date": "2025-03-15", "libelle": "VIREMENT COMPTE TIERS INCONNU",
            "categorie": "658 - Charges diverses", "type": "charge",
            "montant": 2_500_000, "anomalie": 1,
        },
        {
            "date": "2025-07-22", "libelle": "ACHAT HORS FOURNISSEUR HABITUEL",
            "categorie": "601 - Achats marchandises", "type": "charge",
            "montant": 1_800_000, "anomalie": 1,
        },
        {
            "date": "2025-10-05", "libelle": "REMBOURSEMENT INEXPLIQUE",
            "categorie": "658 - Charges diverses", "type": "charge",
            "montant": 980_000, "anomalie": 1,
        },
        {
            "date": "2025-11-30", "libelle": "DOUBLE PAIEMENT LOYER NOVEMBRE",
            "categorie": "631 - Loyer local commercial", "type": "charge",
            "montant": 300_000, "anomalie": 1,
        },
    ]
    rows.extend(anomalies)
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("  Génération du dataset — PME sénégalaise 2025")
    print("  Groupe 17 — Hackathon ISM 2026")
    print("=" * 60)

    transactions = generer_transactions()
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "id", df.index + 1)

    # Sauvegarde
    output = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "transactions_pme.csv"
    )
    output = os.path.normpath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    # Résumé
    print(f"\n[OK] {len(df)} transactions generees")
    print(f"[OK] {df['anomalie'].sum()} transactions anormales (pour Isolation Forest)")
    print(f"[OK] Periode : {df['date'].min().date()} - {df['date'].max().date()}")
    print(f"[OK] Fichier : {output}")

    print("\n--- Répartition par catégorie ---")
    resume = (
        df.groupby(["type", "categorie"])["montant"]
        .agg(nb="count", total="sum", moyenne="mean")
        .reset_index()
    )
    resume["total"] = resume["total"].apply(lambda x: f"{x:,.0f} FCFA")
    resume["moyenne"] = resume["moyenne"].apply(lambda x: f"{x:,.0f} FCFA")
    print(resume.to_string(index=False))

    print("\n--- Aperçu des 5 premières lignes ---")
    print(df.head().to_string(index=False))
