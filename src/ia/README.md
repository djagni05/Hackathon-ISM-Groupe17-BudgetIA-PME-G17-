# Module IA — Classification & Détection d'Anomalies

Ce dossier contient les modules d'intelligence artificielle du projet BudgetIA PME.

## Contenu

| Fichier | Description |
|---------|-------------|
| `generate_dataset.py` | Génération du dataset simulé (612 transactions PME sénégalaise) |
| `classifier.py` | Pipeline TF-IDF + Random Forest — classification OHADA |

## Fonctionnalités

### Classification (Module 1)
- Vectorisation TF-IDF des libellés de transactions
- Classifieur Random Forest — **98.4% de précision**
- 10 catégories OHADA : 601, 621, 624, 626, 627, 631, 641, 658, 701, 706
- Score de confiance et top-3 des catégories possibles

### Détection d'Anomalies (Module 2)
- Isolation Forest (apprentissage non-supervisé)
- Détection de 5 anomalies sur 612 transactions (0.8%)
- Score d'anomalie par transaction

## Dataset

612 transactions simulées avec saisonnalité sénégalaise :
- Tabaski ×2.1 | Korité ×1.9 | Ramadan ×1.4

> Les fichiers de données et modèles sont exclus du dépôt pour des raisons de confidentialité.
