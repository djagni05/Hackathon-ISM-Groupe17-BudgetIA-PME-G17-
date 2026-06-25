# Données — BudgetIA PME

Ce dossier contient les données utilisées par l'application.

## Contenu

| Fichier | Description |
|---------|-------------|
| `transactions_pme.csv` | Dataset de 612 transactions PME sénégalaise simulées |
| `budgetia.db` | Base de données utilisateurs (SQLite, chiffrée Fernet) |
| `journal_audit.db` | Journal d'audit blockchain SHA-256 (SQLite) |
| `.fernet_key` | Clé de chiffrement AES-128-CBC |

## Structure du CSV

```
date, libelle, montant, type, categorie, anomalie
```

> Tous les fichiers de ce dossier sont exclus du depot GitHub pour des raisons de securite et confidentialite.
