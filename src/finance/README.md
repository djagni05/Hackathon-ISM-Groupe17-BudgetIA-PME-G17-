# Module Finance — Budget Prévisionnel

Ce dossier contient le module de prévision budgétaire basé sur **Prophet (Meta)**.

## Contenu

| Fichier | Description |
|---------|-------------|
| `budget_previsionnel.py` | Entraînement et chargement des modèles Prophet |

## Fonctionnalités

- Prévision des produits, charges et solde net sur 15 à 90 jours
- Modélisation de la saisonnalité (Tabaski, Korité, Ramadan)
- Calcul des indicateurs mensuels et tendance trimestrielle
- Intervalles de confiance à 90%

## Modèles générés

- `prophet_produits.pkl`
- `prophet_charges.pkl`
- `prophet_solde_net.pkl`

> Les fichiers `.pkl` sont exclus du dépôt pour des raisons de confidentialité.
