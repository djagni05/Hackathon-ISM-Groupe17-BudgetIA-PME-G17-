"""
Point d'entrée Streamlit Cloud — BudgetIA PME Groupe 17
"""
import sys
import os

# Racine du projet
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Sur Streamlit Cloud, /mount/src/ est en lecture seule.
# On redirige les fichiers écrits (BDD, modèles, clé) vers /tmp
TMP = "/tmp/budgetia"
os.environ.setdefault("BUDGETIA_DATA_DIR",   os.path.join(TMP, "data"))
os.environ.setdefault("BUDGETIA_MODELS_DIR", os.path.join(TMP, "models"))

os.makedirs(os.environ["BUDGETIA_DATA_DIR"],   exist_ok=True)
os.makedirs(os.environ["BUDGETIA_MODELS_DIR"], exist_ok=True)

# Copier le CSV depuis le repo (lecture seule) vers /tmp (écriture)
src_csv = os.path.join(ROOT, "data", "transactions_pme.csv")
dst_csv = os.path.join(os.environ["BUDGETIA_DATA_DIR"], "transactions_pme.csv")
if not os.path.exists(dst_csv) and os.path.exists(src_csv):
    import shutil
    shutil.copy2(src_csv, dst_csv)

# Lancer l'interface principale
import runpy
runpy.run_path(os.path.join(ROOT, "src", "interface", "app.py"), run_name="__main__")
