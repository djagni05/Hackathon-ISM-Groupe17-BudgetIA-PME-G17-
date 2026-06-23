"""
Module Cybersécurité — Protection de la plateforme BudgetIA PME
  1. Chiffrement Fernet des données financières (AES-128-CBC)
  2. Contrôle d'accès par rôle (RBAC)
  3. Journal d'audit horodaté et infalsifiable (chaîne SHA-256)

Hackathon ISM 2026 — Groupe 17
Auteurs : NDZIENGOMO DJAGNI Branham & Sy Kadiatou (Cybersécurité)
"""

import os
import json
import hashlib
import hmac
import sqlite3
import bcrypt
from datetime import datetime
from cryptography.fernet import Fernet

# ── Chemins ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH     = os.path.join(BASE_DIR, "data", "budgetia.db")
KEY_PATH    = os.path.join(BASE_DIR, "data", ".fernet_key")   # NE PAS VERSIONNER
JOURNAL_DB  = os.path.join(BASE_DIR, "data", "journal_audit.db")

# ── Rôles et permissions ─────────────────────────────────────────────
ROLES = {
    "dirigeant":        ["lire", "ecrire", "supprimer", "exporter", "admin"],
    "comptable":        ["lire", "ecrire", "exporter"],
    "expert_comptable": ["lire", "exporter"],
}


# ════════════════════════════════════════════════════════════════════
#  1. CHIFFREMENT FERNET
# ════════════════════════════════════════════════════════════════════

def generer_cle():
    """Génère et sauvegarde une clé Fernet. À faire une seule fois."""
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    cle = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(cle)
    return cle


def charger_cle():
    """Charge la clé Fernet : depuis st.secrets (Streamlit Cloud) ou depuis le fichier local."""
    try:
        import streamlit as st
        if "FERNET_KEY" in st.secrets:
            return st.secrets["FERNET_KEY"].encode()
    except Exception:
        pass
    if not os.path.exists(KEY_PATH):
        return generer_cle()
    with open(KEY_PATH, "rb") as f:
        return f.read()


def chiffrer(texte: str) -> bytes:
    """Chiffre une chaîne de caractères avec Fernet (AES-128)."""
    cle = charger_cle()
    f   = Fernet(cle)
    return f.encrypt(texte.encode("utf-8"))


def dechiffrer(token: bytes) -> str:
    """Déchiffre un token Fernet."""
    cle = charger_cle()
    f   = Fernet(cle)
    return f.decrypt(token).decode("utf-8")


def chiffrer_csv(chemin_csv: str, chemin_sortie: str):
    """Chiffre un fichier CSV entier."""
    with open(chemin_csv, "r", encoding="utf-8-sig") as fich:
        contenu = fich.read()
    chiffre = chiffrer(contenu)
    with open(chemin_sortie, "wb") as fich:
        fich.write(chiffre)
    return chemin_sortie


def dechiffrer_csv(chemin_chiffre: str) -> str:
    """Déchiffre un fichier CSV chiffré et retourne le contenu."""
    with open(chemin_chiffre, "rb") as fich:
        token = fich.read()
    return dechiffrer(token)


# ════════════════════════════════════════════════════════════════════
#  2. GESTION DES UTILISATEURS ET RÔLES (RBAC)
# ════════════════════════════════════════════════════════════════════

def _init_db():
    """Initialise la base SQLite pour les utilisateurs."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS utilisateurs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            pwd_hash  TEXT NOT NULL,
            role      TEXT NOT NULL,
            actif     INTEGER DEFAULT 1,
            cree_le   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def creer_utilisateur(username: str, mot_de_passe: str, role: str):
    """
    Crée un utilisateur avec son mot de passe haché (bcrypt).
    Le mot de passe n'est JAMAIS stocké en clair.
    """
    if role not in ROLES:
        raise ValueError(f"Role invalide. Choisir parmi : {list(ROLES.keys())}")

    _init_db()
    pwd_hash = bcrypt.hashpw(mot_de_passe.encode(), bcrypt.gensalt()).decode()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO utilisateurs (username, pwd_hash, role, cree_le) VALUES (?, ?, ?, ?)",
            (username, pwd_hash, role, datetime.now().isoformat())
        )
        conn.commit()
        journaliser("SYSTEME", "creation_utilisateur", f"Nouvel utilisateur : {username} ({role})")
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authentifier(username: str, mot_de_passe: str):
    """
    Vérifie les identifiants. Retourne (True, role) ou (False, None).
    """
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT pwd_hash, role, actif FROM utilisateurs WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if row is None:
        journaliser(username, "echec_connexion", "Utilisateur inconnu")
        return False, None

    pwd_hash, role, actif = row
    if not actif:
        journaliser(username, "echec_connexion", "Compte desactive")
        return False, None

    if bcrypt.checkpw(mot_de_passe.encode(), pwd_hash.encode()):
        journaliser(username, "connexion_reussie", f"Role : {role}")
        return True, role
    else:
        journaliser(username, "echec_connexion", "Mot de passe incorrect")
        return False, None


def verifier_permission(role: str, action: str) -> bool:
    """Vérifie si un rôle a le droit d'effectuer une action."""
    return action in ROLES.get(role, [])


# ════════════════════════════════════════════════════════════════════
#  3. JOURNAL D'AUDIT INFALSIFIABLE (chaîne de hachage SHA-256)
# ════════════════════════════════════════════════════════════════════

def _init_journal():
    """Initialise la base SQLite du journal d'audit."""
    os.makedirs(os.path.dirname(JOURNAL_DB), exist_ok=True)
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            horodatage    TEXT NOT NULL,
            utilisateur   TEXT NOT NULL,
            action        TEXT NOT NULL,
            details       TEXT,
            hash_entree   TEXT NOT NULL,
            hash_precedent TEXT
        )
    """)
    conn.commit()
    conn.close()


def _dernier_hash() -> str:
    """Retourne le hash de la dernière entrée du journal."""
    _init_journal()
    conn = sqlite3.connect(JOURNAL_DB)
    row  = conn.execute(
        "SELECT hash_entree FROM journal ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else "GENESIS"


def _calculer_hash(horodatage, utilisateur, action, details, hash_precedent) -> str:
    """Calcule le SHA-256 d'une entrée du journal (chainé)."""
    contenu = f"{horodatage}|{utilisateur}|{action}|{details}|{hash_precedent}"
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()


def journaliser(utilisateur: str, action: str, details: str = ""):
    """
    Ajoute une entrée dans le journal d'audit avec chaining SHA-256.
    Chaque entrée contient le hash de l'entrée précédente,
    rendant toute modification détectable.
    """
    _init_journal()
    horodatage     = datetime.now().isoformat(timespec="seconds")
    hash_precedent = _dernier_hash()
    hash_entree    = _calculer_hash(horodatage, utilisateur, action, details, hash_precedent)

    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute(
        """INSERT INTO journal
           (horodatage, utilisateur, action, details, hash_entree, hash_precedent)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (horodatage, utilisateur, action, details, hash_entree, hash_precedent)
    )
    conn.commit()
    conn.close()


def verifier_integrite_journal() -> tuple[bool, list]:
    """
    Vérifie que la chaîne de hachage du journal est intacte.
    Retourne (integre: bool, anomalies: list)
    """
    _init_journal()
    conn = sqlite3.connect(JOURNAL_DB)
    entrees = conn.execute(
        "SELECT id, horodatage, utilisateur, action, details, hash_entree, hash_precedent "
        "FROM journal ORDER BY id ASC"
    ).fetchall()
    conn.close()

    anomalies = []
    hash_attendu = "GENESIS"

    for entree in entrees:
        id_, horodatage, utilisateur, action, details, hash_reel, hash_prec = entree

        # Vérifier que le hash_precedent correspond
        if hash_prec != hash_attendu:
            anomalies.append(f"Entree #{id_} : hash precedent incorrect (falsification detectee)")

        # Recalculer le hash de cette entrée
        hash_calc = _calculer_hash(horodatage, utilisateur, action, details or "", hash_prec)
        if hash_calc != hash_reel:
            anomalies.append(f"Entree #{id_} : hash invalide (contenu modifie)")

        hash_attendu = hash_reel

    return len(anomalies) == 0, anomalies


def lire_journal(limit: int = 50) -> list:
    """Retourne les N dernières entrées du journal."""
    _init_journal()
    conn = sqlite3.connect(JOURNAL_DB)
    rows = conn.execute(
        "SELECT id, horodatage, utilisateur, action, details, hash_entree "
        "FROM journal ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "horodatage": r[1], "utilisateur": r[2],
         "action": r[3], "details": r[4], "hash": r[5][:12] + "..."}
        for r in rows
    ]


# ════════════════════════════════════════════════════════════════════
#  SCRIPT PRINCIPAL — Démonstration
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE CYBERSECURITE — BudgetIA PME")
    print("  Groupe 17 — Hackathon ISM 2026")
    print("=" * 60)

    # 1. Chiffrement
    print("\n--- 1. CHIFFREMENT FERNET ---")
    message = "Ventes journalieres : 245,000 FCFA - CONFIDENTIEL"
    chiffre = chiffrer(message)
    dechiffre = dechiffrer(chiffre)
    print(f"Original  : {message}")
    print(f"Chiffre   : {chiffre[:50]}...")
    print(f"Dechiffre : {dechiffre}")
    print(f"Integrite : {'OK' if message == dechiffre else 'ERREUR'}")

    # 2. Utilisateurs et rôles
    print("\n--- 2. GESTION DES ROLES ---")
    creer_utilisateur("admin_pme",    "MotDePasse@2026", "dirigeant")
    creer_utilisateur("comptable1",   "Compta@Secure1",  "comptable")
    creer_utilisateur("expert_ext",   "Expert@OHADA1",   "expert_comptable")

    for user, pwd in [("admin_pme", "MotDePasse@2026"), ("comptable1", "mauvais_mdp")]:
        ok, role = authentifier(user, pwd)
        print(f"Connexion '{user}' : {'AUTORISEE (' + role + ')' if ok else 'REFUSEE'}")

    print("\nPermissions par role :")
    for role, perms in ROLES.items():
        print(f"  {role:<20} : {', '.join(perms)}")

    # 3. Journal d'audit
    print("\n--- 3. JOURNAL D'AUDIT ---")
    journaliser("admin_pme",  "import_csv",        "Fichier : releve_jan2025.csv (612 lignes)")
    journaliser("comptable1", "classification",     "Transaction #42 classee : 631 - Loyer")
    journaliser("admin_pme",  "export_pdf",         "Rapport mensuel janvier 2025")
    journaliser("expert_ext", "consultation",       "Lecture etats financiers 2025")

    integre, anomalies = verifier_integrite_journal()
    print(f"Integrite du journal : {'CONFIRMEE' if integre else 'COMPROMISE'}")
    if anomalies:
        for a in anomalies:
            print(f"  [ALERTE] {a}")

    print("\nDernieres entrees du journal :")
    for entree in lire_journal(5):
        print(f"  [{entree['horodatage']}] {entree['utilisateur']:<15} | "
              f"{entree['action']:<25} | hash:{entree['hash']}")
