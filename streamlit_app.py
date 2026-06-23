"""
Point d'entrée Streamlit Cloud — BudgetIA PME Groupe 17
Redirige vers l'interface principale dans src/interface/app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Importer l'app principale (exécute tout le code Streamlit)
exec(open(os.path.join(os.path.dirname(__file__), "src", "interface", "app.py"), encoding="utf-8").read())
