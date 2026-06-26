"""
BudgetIA PME — Interface principale Streamlit
Plateforme de budgétisation et comptabilité automatisée (OHADA)
Hackathon ISM 2026 — Groupe 17
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Ajout des chemins des modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.cyber.securite import (
    authentifier, creer_utilisateur, verifier_permission,
    journaliser, lire_journal, verifier_integrite_journal, chiffrer,
)
from src.ia.classifier import (
    entrainer_classifieur, charger_classifieur,
    classifier_transaction, entrainer_isolation_forest,
)
from src.finance.budget_previsionnel import (
    charger_et_preparer, obtenir_previsions,
    resume_previsions, calculer_indicateurs,
)

# Chemins : utilise /tmp sur Streamlit Cloud, sinon le dossier du repo
_DATA_DIR   = os.environ.get("BUDGETIA_DATA_DIR",   os.path.join(ROOT, "data"))
_MODELS_DIR = os.environ.get("BUDGETIA_MODELS_DIR", os.path.join(ROOT, "models"))
DATA_PATH   = os.path.join(_DATA_DIR,   "transactions_pme.csv")
MODELS_PATH = _MODELS_DIR


# ────────────────────────────────────────────────────────────────────
#  INITIALISATION AU DÉMARRAGE (comptes démo, dossiers)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialiser_app():
    from src.cyber.securite import creer_utilisateur, _init_db
    _init_db()
    comptes = [
        ("admin_pme",  "BudgetIA@2026",  "dirigeant"),
        ("comptable1", "Compta@PME2026", "comptable"),
        ("expert_ext", "Expert@OHADA26", "expert_comptable"),
    ]
    for username, pwd, role in comptes:
        try:
            creer_utilisateur(username, pwd, role)
        except Exception:
            pass

initialiser_app()

# ────────────────────────────────────────────────────────────────────
#  CONFIG PAGE
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BudgetIA PME",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
    /* ══════════════════════════════════════════
       THÈME SOMBRE — BudgetIA PME
    ══════════════════════════════════════════ */

    /* ── Fond général ── */
    .stApp {
        background: #0f1929 !important;
    }
    .block-container {
        background: transparent !important;
        padding: 2rem 3rem 3rem 3rem !important;
        max-width: 1200px !important;
    }

    /* ── Texte général ── */
    .stApp, .stApp p, .stApp span, .stApp div,
    .stApp label, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4, h5 {
        color: #93c5fd !important;
    }

    /* ── Page login ── */
    .login-wrapper {
        background: #1a2d4a;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        border-top: 6px solid #3b82f6;
    }
    .login-logo {
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .login-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #93c5fd !important;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .login-subtitle {
        font-size: 0.95rem;
        color: #94a3b8 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .login-badge {
        display: inline-block;
        background: #243654;
        color: #93c5fd !important;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 2px;
    }
    .login-footer {
        text-align: center;
        color: #64748b !important;
        font-size: 0.78rem;
        margin-top: 1.2rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f36 0%, #0a1628 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 4px;
        display: block;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(59,130,246,0.25);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12) !important;
    }

    /* ── En-têtes de page ── */
    .page-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 1.4rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid #2d5a9e;
    }
    .page-header h1 {
        color: #ffffff !important;
        font-size: 1.7rem;
        margin: 0 0 0.2rem 0;
    }
    .page-header p {
        color: rgba(255,255,255,0.70) !important;
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Cartes de section ── */
    .section-card {
        background: #1a2d4a;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        border: 1px solid #2d4a6e;
    }
    .section-card, .section-card * {
        color: #e2e8f0 !important;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #93c5fd !important;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }

    /* ── KPI Cards ── */
    [data-testid="metric-container"] {
        background: #1a2d4a !important;
        border-radius: 14px;
        padding: 1.3rem 1.2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        border: 1px solid #2d4a6e !important;
        border-left: 5px solid #3b82f6 !important;
        margin-bottom: 0.5rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.82rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #93c5fd !important;
    }

    /* ── Alertes ── */
    .alerte-rouge {
        background: #2d1515;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #fca5a5 !important;
    }
    .alerte-verte {
        background: #152d1e;
        border-left: 4px solid #22c55e;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        color: #86efac !important;
    }
    .alerte-orange {
        background: #2d1f0f;
        border-left: 4px solid #f97316;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #fdba74 !important;
    }

    /* ── Hash badge audit ── */
    .hash-badge {
        font-family: monospace;
        font-size: 0.70rem;
        background: #0f1929;
        color: #60a5fa !important;
        padding: 3px 8px;
        border-radius: 5px;
        border: 1px solid #2d4a6e;
        letter-spacing: 0.5px;
    }

    /* ── Inputs ── */
    .stTextInput input, input[type="text"], input[type="password"] {
        background: #243654 !important;
        color: #e2e8f0 !important;
        border: 1.5px solid #3d5a80 !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder {
        color: #64748b !important;
    }
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #243654 !important;
        color: #e2e8f0 !important;
        border: 1.5px solid #3d5a80 !important;
    }

    /* ── Boutons ── */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        opacity: 0.88;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1.2rem;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        color: #93c5fd !important;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border: 1px solid #2d4a6e;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #2d4a6e;
        margin: 1.5rem 0;
    }

    /* ── Résultat classification ── */
    .result-card {
        background: #1a2d4a;
        border-radius: 14px;
        padding: 1.5rem;
        border: 2px solid #22c55e;
        margin-top: 1rem;
        color: #e2e8f0 !important;
    }

    /* ── Colonnes ── */
    div[data-testid="column"] { padding: 0 0.6rem; }
    .stSlider { padding: 0.5rem 0 1rem 0; }
    .stFileUploader { padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────────────
if "connecte"    not in st.session_state: st.session_state.connecte    = False
if "username"    not in st.session_state: st.session_state.username    = ""
if "role"        not in st.session_state: st.session_state.role        = ""
if "classifieur" not in st.session_state: st.session_state.classifieur = None
if "iso_forest"  not in st.session_state: st.session_state.iso_forest  = None
if "df"          not in st.session_state: st.session_state.df          = None
if "csv_path"    not in st.session_state: st.session_state.csv_path    = None


# ────────────────────────────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (une seule fois)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def charger_modeles(csv_path):
    clf_path = os.path.join(MODELS_PATH, "classifieur_categories.pkl")
    iso_path = os.path.join(MODELS_PATH, "isolation_forest.pkl")

    if os.path.exists(clf_path):
        clf = charger_classifieur()
    else:
        with st.spinner("Entrainement du classifieur IA..."):
            clf, _, _ = entrainer_classifieur(csv_path)

    if os.path.exists(iso_path):
        import pickle
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    else:
        with st.spinner("Entrainement Isolation Forest..."):
            iso, _ = entrainer_isolation_forest(csv_path)

    return clf, iso


@st.cache_data(show_spinner=False)
def charger_previsions(csv_path):
    return obtenir_previsions(csv_path, horizon_jours=90)


# ────────────────────────────────────────────────────────────────────
#  PAGE IMPORT DONNÉES
# ────────────────────────────────────────────────────────────────────
def page_import_donnees():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="page-header" style="text-align:center;">
            <h1>📂 Importer vos données</h1>
            <p>Chargez votre fichier CSV de transactions pour démarrer l'analyse</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Format attendu du fichier CSV</div>', unsafe_allow_html=True)
        st.markdown("""
| Colonne | Type | Exemple |
|---------|------|---------|
| `date` | YYYY-MM-DD | 2025-01-15 |
| `libelle` | texte | FACTURE SENELEC |
| `montant` | nombre | 125000 |
| `type` | charge / produit | charge |
| `categorie` | code OHADA | 626 - Frais postaux |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Charger votre fichier</div>', unsafe_allow_html=True)

        fichier = st.file_uploader("Sélectionner un fichier CSV", type=["csv"], label_visibility="collapsed")

        if fichier is not None:
            try:
                df = pd.read_csv(fichier)
                df["date"] = pd.to_datetime(df["date"])

                colonnes_requises = ["date", "libelle", "montant", "type"]
                manquantes = [c for c in colonnes_requises if c not in df.columns]

                if manquantes:
                    st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
                else:
                    # Sauvegarder le CSV dans le dossier data
                    csv_dest = os.path.join(_DATA_DIR, "transactions_import.csv")
                    df.to_csv(csv_dest, index=False)

                    st.session_state.df       = df
                    st.session_state.csv_path = csv_dest

                    st.success(f"✅ {len(df)} transactions chargées avec succès !")
                    st.markdown(f"**Aperçu des données :**")
                    st.dataframe(df.head(5), use_container_width=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Lancer l'analyse →", use_container_width=True):
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE LOGIN
# ────────────────────────────────────────────────────────────────────
def page_login():
    # Centrage vertical
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.6, 1])

    with col2:
        st.markdown("""
        <div class="login-wrapper">
            <div class="login-logo">💼</div>
            <div class="login-title">BudgetIA PME</div>
            <div class="login-subtitle">Plateforme de comptabilité automatisée — Normes OHADA</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤  Nom d'utilisateur", placeholder="ex: admin_pme")
            password = st.text_input("🔒  Mot de passe", type="password", placeholder="••••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Se connecter →", use_container_width=True)

        if submit:
            if not username or not password:
                st.error("Veuillez remplir tous les champs.")
            else:
                ok, role = authentifier(username, password)
                if ok:
                    st.session_state.connecte = True
                    st.session_state.username = username
                    st.session_state.role     = role
                    st.success(f"Bienvenue {username} ! Rôle : {role}")
                    st.rerun()
                else:
                    st.error("Identifiants incorrects ou compte inactif.")

        st.markdown("""
        <div class="login-footer">
            <span class="login-badge">🔐 AES-128</span>
            <span class="login-badge">🛡 RBAC</span>
            <span class="login-badge">📋 OHADA</span>
            <br><br>
            Groupe 17 · Hackathon ISM 2026 · Cybersécurité + MOSIEF
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size:2.2rem;">💼</div>
            <div style="font-size:1.3rem; font-weight:800; letter-spacing:0.5px;">BudgetIA PME</div>
            <div style="font-size:0.75rem; opacity:0.7; margin-top:2px;">Hackathon ISM 2026 · Groupe 17</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        role_icons = {
            "dirigeant":        "👑",
            "comptable":        "📊",
            "expert_comptable": "🔍",
        }
        icon = role_icons.get(st.session_state.role, "👤")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.12); border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem;">
            <div style="font-size:0.8rem; opacity:0.7;">Connecté en tant que</div>
            <div style="font-size:1rem; font-weight:700;">{icon}  {st.session_state.username}</div>
            <div style="font-size:0.78rem; opacity:0.8; margin-top:2px;">Rôle : {st.session_state.role.replace('_',' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Navigation**")
        pages = [
            "📈  Tableau de bord",
            "🤖  Classification",
            "📅  Budget prévisionnel",
            "🚨  Anomalies",
        ]
        if verifier_permission(st.session_state.role, "admin"):
            pages.append("📋  Journal d'audit")

        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("---")

        # Infos fichier chargé
        if st.session_state.df is not None:
            nb = len(st.session_state.df)
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.15); border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.8rem; font-size:0.8rem;">
                📂 <strong>{nb} transactions</strong> chargées
            </div>
            """, unsafe_allow_html=True)
            if st.button("📂  Changer de fichier", use_container_width=True):
                st.session_state.df       = None
                st.session_state.csv_path = None
                st.rerun()

        if st.button("🚪  Déconnexion", use_container_width=True):
            journaliser(st.session_state.username, "deconnexion", "")
            for key in ["connecte", "username", "role", "df", "csv_path"]:
                st.session_state[key] = "" if key not in ["connecte", "df", "csv_path"] else (False if key == "connecte" else None)
            st.rerun()

        st.markdown("""
        <div style="text-align:center; opacity:0.45; font-size:0.72rem; margin-top:1rem;">
            🔐 Données chiffrées AES-128<br>
            📋 Normes OHADA / SYSCOHADA
        </div>
        """, unsafe_allow_html=True)

    return page


# ────────────────────────────────────────────────────────────────────
#  PAGE 1 : TABLEAU DE BORD
# ────────────────────────────────────────────────────────────────────
def page_dashboard(df):
    journaliser(st.session_state.username, "consultation", "Tableau de bord")

    st.markdown("""
    <div class="page-header">
        <h1>📈 Tableau de bord financier</h1>
        <p>Vue d'ensemble de la santé financière de votre PME — Année 2025</p>
    </div>
    """, unsafe_allow_html=True)

    charges      = df[df["type"] == "charge"]["montant"].sum()
    produits     = df[df["type"] == "produit"]["montant"].sum()
    solde        = produits - charges
    nb_anomalies = df[df["anomalie"] == 1]["anomalie"].sum() if "anomalie" in df.columns else 0

    # ── KPIs ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Produits",  f"{produits:,.0f} FCFA",  delta="Année 2025")
    c2.metric("📉 Total Charges",   f"{charges:,.0f} FCFA")
    c3.metric("📊 Résultat Net",    f"{solde:,.0f} FCFA",
              delta="Bénéfice" if solde > 0 else "Déficit",
              delta_color="normal" if solde > 0 else "inverse")
    c4.metric("🚨 Alertes Fraude",  f"{int(nb_anomalies)} transaction(s)",
              delta="À vérifier" if nb_anomalies > 0 else "Aucune alerte",
              delta_color="inverse" if nb_anomalies > 0 else "off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques ────────────────────────────────────────────────────
    col_g, col_d = st.columns([3, 2], gap="large")

    with col_g:
        st.markdown('<div class="section-card"><div class="section-title">Évolution mensuelle — Produits vs Charges</div>', unsafe_allow_html=True)
        df_m = df.copy()
        df_m["mois"] = df_m["date"].dt.to_period("M").astype(str)
        mensuel = df_m.groupby(["mois", "type"])["montant"].sum().reset_index()
        fig = px.bar(
            mensuel, x="mois", y="montant", color="type",
            barmode="group",
            color_discrete_map={"produit": "#22c55e", "charge": "#ef4444"},
            labels={"montant": "FCFA", "mois": "", "type": ""},
        )
        fig.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="section-card"><div class="section-title">Répartition des charges</div>', unsafe_allow_html=True)
        charges_df = df[df["type"] == "charge"].groupby("categorie")["montant"].sum().reset_index()
        charges_df["categorie"] = charges_df["categorie"].str.split(" - ").str[1]
        fig2 = px.pie(
            charges_df, values="montant", names="categorie",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig2.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="v", font_size=10),
            paper_bgcolor="white",
        )
        fig2.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Solde cumulé ─────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Évolution du solde cumulé sur l\'année</div>', unsafe_allow_html=True)
    df_daily = df.groupby(["date", "type"])["montant"].sum().unstack(fill_value=0).reset_index()
    if "produit" not in df_daily: df_daily["produit"] = 0
    if "charge"  not in df_daily: df_daily["charge"]  = 0
    df_daily["solde_net"]    = df_daily["produit"] - df_daily["charge"]
    df_daily["solde_cumule"] = df_daily["solde_net"].cumsum()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_daily["date"], y=df_daily["solde_cumule"],
        fill="tozeroy", name="Solde cumulé",
        line=dict(color="#2674B5", width=2.5),
        fillcolor="rgba(38,116,181,0.1)",
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig3.update_layout(
        height=260, margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="FCFA", xaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE 2 : CLASSIFICATION
# ────────────────────────────────────────────────────────────────────
def page_classification(clf):
    if not verifier_permission(st.session_state.role, "lire"):
        st.error("Accès refusé.")
        return

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Classification automatique OHADA</h1>
        <p>Catégorisation intelligente des transactions par IA (TF-IDF + Random Forest — 98.4% précision)</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✏️ Saisie manuelle", "📂 Importer un CSV"])

    # ── Saisie manuelle ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-card"><div class="section-title">Saisir une transaction</div>', unsafe_allow_html=True)
        st.markdown("Entrez le libellé d'une transaction pour la classifier automatiquement selon les normes OHADA.")
        libelle = st.text_input("Libellé de la transaction",
                                placeholder="ex: FACTURE SENELEC ELECTRICITE")

        if st.button("Classifier", key="btn_class") and libelle:
            if not verifier_permission(st.session_state.role, "ecrire"):
                st.warning("Votre role ne permet pas d'effectuer des classifications.")
            else:
                cat, confiance, top3 = classifier_transaction(libelle, clf)
                journaliser(st.session_state.username, "classification", f"'{libelle}' => {cat}")

                col_r, col_t = st.columns(2)
                with col_r:
                    couleur = "#22c55e" if confiance > 80 else "#f59e0b" if confiance > 50 else "#ef4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Catégorie détectée</strong><br>
                        <span style="font-size:1.2rem; color:{couleur};">{cat}</span><br>
                        <span>Confiance : <strong>{confiance}%</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_t:
                    st.markdown("**Top 3 catégories possibles :**")
                    for c, p in top3:
                        barre = int(p / 5)
                        st.markdown(f"`{c[:35]}` {'█' * barre} {p}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Import CSV ───────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-card"><div class="section-title">Importer un relevé CSV</div>', unsafe_allow_html=True)
        st.markdown("Importez un relevé bancaire CSV avec une colonne `libelle`.")
        fichier = st.file_uploader("Choisir un fichier CSV", type=["csv"])

        if fichier:
            df_import = pd.read_csv(fichier)
            st.dataframe(df_import.head(), use_container_width=True)

            if "libelle" not in df_import.columns:
                st.error("Le fichier doit contenir une colonne 'libelle'.")
            else:
                if st.button("Classifier toutes les transactions"):
                    with st.spinner("Classification en cours..."):
                        resultats = []
                        for lib in df_import["libelle"]:
                            cat, conf, _ = classifier_transaction(str(lib), clf)
                            resultats.append({"libelle": lib, "categorie": cat, "confiance_%": conf})
                        df_res = pd.DataFrame(resultats)
                        journaliser(
                            st.session_state.username, "import_csv",
                            f"{len(df_res)} transactions classifiees"
                        )
                    st.success(f"{len(df_res)} transactions classifiées !")
                    st.dataframe(df_res, use_container_width=True)

                    csv_out = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Télécharger le résultat",
                        csv_out,
                        "transactions_classifiees.csv",
                        "text/csv",
                    )
        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE 3 : BUDGET PRÉVISIONNEL
# ────────────────────────────────────────────────────────────────────
def page_budget(csv_path):
    journaliser(st.session_state.username, "consultation", "Budget previsionnel")

    st.markdown("""
    <div class="page-header">
        <h1>📅 Budget prévisionnel</h1>
        <p>Prévisions financières basées sur Prophet (Meta) — Modélisation avec saisonnalité</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Paramètres de prévision</div>', unsafe_allow_html=True)
    horizon = st.slider("Horizon de prévision (jours)", 15, 90, 30, 15)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Calcul des prévisions en cours..."):
        resultats = charger_previsions(csv_path)
        resume    = resume_previsions(resultats, horizon_jours=horizon)
        mensuel, tendance = calculer_indicateurs(csv_path)

    # ── KPIs prévisions ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Produits prévus",
              f"{resume['produits']['total_prevu']:,.0f} FCFA",
              f"±{(resume['produits']['intervalle_haut'] - resume['produits']['intervalle_bas'])//2:,.0f}")
    c2.metric("📉 Charges prévues",
              f"{resume['charges']['total_prevu']:,.0f} FCFA",
              f"±{(resume['charges']['intervalle_haut'] - resume['charges']['intervalle_bas'])//2:,.0f}")
    solde_prevu = resume['solde_net']['total_prevu']
    c3.metric("📊 Solde prévu",
              f"{solde_prevu:,.0f} FCFA",
              "Excédent" if solde_prevu > 0 else "Déficit",
              delta_color="normal" if solde_prevu > 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphique prévisions vs historique ─────────────────────────
    col_g, col_d = st.columns([3, 2], gap="large")

    with col_g:
        st.markdown('<div class="section-card"><div class="section-title">' + f'Prévision des produits — horizon {horizon} jours</div>', unsafe_allow_html=True)
        data_prod = resultats["produits"]
        hist = data_prod["historique"].rename(columns={"ds": "date", "reel": "valeur"})
        hist["type"] = "Historique"
        fc   = data_prod["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon + 30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["valeur"],
            name="Historique", line=dict(color="#3b82f6", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            name="Prévision", line=dict(color="#22c55e", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(34,197,94,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalle 90%",
        ))
        fig.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="section-card"><div class="section-title">Résultats mensuels (6 derniers mois)</div>', unsafe_allow_html=True)
        df_m = mensuel[["produit", "charge", "resultat_net", "taux_marge"]].tail(6).reset_index()
        df_m["mois"] = df_m["mois"].astype(str)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_m["mois"], y=df_m["produit"],   name="Produits", marker_color="#22c55e"))
        fig2.add_trace(go.Bar(x=df_m["mois"], y=df_m["charge"],    name="Charges",  marker_color="#ef4444"))
        fig2.add_trace(go.Scatter(
            x=df_m["mois"], y=df_m["resultat_net"],
            name="Résultat net", yaxis="y2",
            line=dict(color="#f59e0b", width=2),
        ))
        fig2.update_layout(
            barmode="group", height=300, margin=dict(t=10, b=10, l=10, r=10),
            yaxis2=dict(overlaying="y", side="right"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tendance_icon = "📈" if tendance == "hausse" else "📉" if tendance == "baisse" else "➡️"
    st.info(f"{tendance_icon} Tendance des 3 derniers mois : **{tendance.upper()}**")


# ────────────────────────────────────────────────────────────────────
#  PAGE 4 : ANOMALIES
# ────────────────────────────────────────────────────────────────────
def page_anomalies(df, iso):
    journaliser(st.session_state.username, "consultation", "Detection anomalies")

    st.markdown("""
    <div class="page-header">
        <h1>🚨 Détection des anomalies</h1>
        <p>Identification des transactions suspectes par Isolation Forest (apprentissage non-supervisé)</p>
    </div>
    """, unsafe_allow_html=True)

    # Préparer les features
    df_feat = df.copy()
    df_feat["mois"]         = df_feat["date"].dt.month
    df_feat["jour_semaine"] = df_feat["date"].dt.dayofweek
    df_feat["est_charge"]   = (df_feat["type"] == "charge").astype(int)

    features = df_feat[["montant", "mois", "jour_semaine", "est_charge"]]
    scores   = iso.decision_function(features)
    preds    = iso.predict(features)

    df_feat["score_anomalie"] = scores
    df_feat["suspect"]        = (preds == -1)

    suspectes  = df_feat[df_feat["suspect"]].sort_values("score_anomalie")
    normales   = df_feat[~df_feat["suspect"]]

    # ── KPIs ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🔍 Transactions analysées", len(df))
    c2.metric("⚠️ Transactions suspectes", len(suspectes),
              delta=f"{len(suspectes)/len(df)*100:.1f}%", delta_color="inverse")
    c3.metric("💸 Montant suspect total",
              f"{suspectes['montant'].sum():,.0f} FCFA")

    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_d = st.columns([3, 2], gap="large")

    with col_g:
        st.markdown('<div class="section-card"><div class="section-title">Transactions suspectes détectées</div>', unsafe_allow_html=True)
        if len(suspectes) > 0:
            df_show = suspectes[["date", "libelle", "categorie", "montant", "score_anomalie"]].copy()
            df_show["date"]   = df_show["date"].dt.strftime("%Y-%m-%d")
            df_show["score"]  = df_show["score_anomalie"].round(4)
            df_show["niveau"] = df_show["score_anomalie"].apply(
                lambda s: "CRITIQUE" if s < -0.05 else "SUSPECT"
            )
            df_show = df_show.drop(columns=["score_anomalie"])

            def colorier(row):
                if row["niveau"] == "CRITIQUE":
                    return ["background-color: #fef2f2"] * len(row)
                return ["background-color: #fffbeb"] * len(row)

            st.dataframe(
                df_show.style.apply(colorier, axis=1),
                use_container_width=True, height=350,
            )
        else:
            st.success("✅ Aucune transaction suspecte détectée.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="section-card"><div class="section-title">Distribution des montants</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=normales["montant"], name="Normal",
            marker_color="#22c55e", opacity=0.7, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=suspectes["montant"], name="Suspect",
            marker_color="#ef4444", opacity=0.9, nbinsx=15,
        ))
        fig.update_layout(
            barmode="overlay", height=350,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Montant (FCFA)",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Alertes critiques ─────────────────────────────────────────────
    critiques = suspectes[suspectes["score_anomalie"] < -0.04]
    if len(critiques) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><div class="section-title">🚨 Alertes critiques</div>', unsafe_allow_html=True)
        for _, row in critiques.iterrows():
            st.markdown(f"""
            <div class="alerte-rouge">
                <strong>🚨 ALERTE CRITIQUE</strong> — {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}<br>
                {row['libelle']} — <strong>{row['montant']:,.0f} FCFA</strong><br>
                Score anomalie : {row['score_anomalie']:.4f}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE 5 : JOURNAL D'AUDIT
# ────────────────────────────────────────────────────────────────────
def page_journal():
    st.markdown("""
    <div class="page-header">
        <h1>📋 Journal d'audit</h1>
        <p>Traçabilité complète des actions — Chaîne de hachage SHA-256 (style blockchain)</p>
    </div>
    """, unsafe_allow_html=True)

    integre, anomalies = verifier_integrite_journal()

    if integre:
        st.markdown("""
        <div class="alerte-verte">
            <strong>✅ INTÉGRITÉ CONFIRMÉE</strong> — La chaîne de hachage SHA-256 est intacte.
            Aucune modification non autorisée détectée.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alerte-rouge">
            <strong>🚨 ALERTE — INTÉGRITÉ COMPROMISE</strong> — Une modification non autorisée a été détectée.
        </div>
        """, unsafe_allow_html=True)
        for a in anomalies:
            st.error(a)

    st.markdown("<br>", unsafe_allow_html=True)
    entrees = lire_journal(100)

    if entrees:
        df_journal = pd.DataFrame(entrees)
        df_journal["hash"] = df_journal["hash"].apply(
            lambda h: f'<span class="hash-badge">{h}</span>'
        )

        # ── Filtres ───────────────────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">Filtres</div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2, gap="large")
        with col_f1:
            users = ["Tous"] + list(df_journal["utilisateur"].unique())
            filtre_user = st.selectbox("👤 Filtrer par utilisateur", users)
        with col_f2:
            actions = ["Toutes"] + list(df_journal["action"].unique())
            filtre_action = st.selectbox("⚡ Filtrer par action", actions)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        df_filtre = df_journal.copy()
        if filtre_user   != "Tous":    df_filtre = df_filtre[df_filtre["utilisateur"] == filtre_user]
        if filtre_action != "Toutes":  df_filtre = df_filtre[df_filtre["action"]      == filtre_action]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"**{len(df_filtre)} entrée(s)** dans le journal")
        st.write(
            df_filtre[["id", "horodatage", "utilisateur", "action", "details", "hash"]]
            .to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Le journal d'audit est vide pour le moment.")


# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.connecte:
        page_login()
        return

    # Si aucune donnée importée, afficher l'écran d'import
    if st.session_state.df is None:
        page_import_donnees()
        return

    df       = st.session_state.df
    csv_path = st.session_state.csv_path
    clf, iso = charger_modeles(csv_path)
    page     = sidebar()

    if "Tableau de bord" in page:
        page_dashboard(df)
    elif "Classification" in page:
        page_classification(clf)
    elif "Budget" in page:
        page_budget(csv_path)
    elif "Anomalies" in page:
        page_anomalies(df, iso)
    elif "Journal" in page:
        page_journal()


if __name__ == "__main__":
    main()
