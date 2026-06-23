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
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .alerte-rouge {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.8rem;
        border-radius: 6px;
    }
    .alerte-verte {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.8rem;
        border-radius: 6px;
    }
    .hash-badge {
        font-family: monospace;
        font-size: 0.75rem;
        background: #1e293b;
        color: #94a3b8;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────────────
if "connecte"   not in st.session_state: st.session_state.connecte   = False
if "username"   not in st.session_state: st.session_state.username   = ""
if "role"       not in st.session_state: st.session_state.role       = ""
if "classifieur" not in st.session_state: st.session_state.classifieur = None
if "iso_forest" not in st.session_state: st.session_state.iso_forest = None


# ────────────────────────────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (une seule fois)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def charger_modeles():
    clf_path = os.path.join(MODELS_PATH, "classifieur_categories.pkl")
    iso_path = os.path.join(MODELS_PATH, "isolation_forest.pkl")

    if os.path.exists(clf_path):
        clf = charger_classifieur()
    else:
        with st.spinner("Entrainement du classifieur IA..."):
            clf, _, _ = entrainer_classifieur(DATA_PATH)

    if os.path.exists(iso_path):
        import pickle
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    else:
        with st.spinner("Entrainement Isolation Forest..."):
            iso, _ = entrainer_isolation_forest(DATA_PATH)

    return clf, iso


@st.cache_data(show_spinner=False)
def charger_donnees():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def charger_previsions():
    return obtenir_previsions(DATA_PATH, horizon_jours=90)


# ────────────────────────────────────────────────────────────────────
#  PAGE LOGIN
# ────────────────────────────────────────────────────────────────────
def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 💼 BudgetIA PME")
        st.markdown("#### Plateforme de comptabilité automatisée — OHADA")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur", placeholder="ex: admin_pme")
            password = st.text_input("Mot de passe", type="password")
            submit   = st.form_submit_button("Se connecter", use_container_width=True)

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

        st.markdown("---")
        st.caption("Comptes de démo — créez-les via `python src/cyber/securite.py`")
        st.caption("Groupe 17 · Hackathon ISM 2026 · Cybersécurité + MOSIEF")


# ────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### 💼 BudgetIA PME")
        st.markdown(f"**Utilisateur :** {st.session_state.username}")
        st.markdown(f"**Rôle :** `{st.session_state.role}`")
        st.markdown("---")

        pages = ["Tableau de bord", "Classification", "Budget previsionnel", "Anomalies"]
        if verifier_permission(st.session_state.role, "admin"):
            pages.append("Journal d'audit")

        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("---")
        if st.button("Deconnexion", use_container_width=True):
            journaliser(st.session_state.username, "deconnexion", "")
            for key in ["connecte", "username", "role"]:
                st.session_state[key] = "" if key != "connecte" else False
            st.rerun()

    return page


# ────────────────────────────────────────────────────────────────────
#  PAGE 1 : TABLEAU DE BORD
# ────────────────────────────────────────────────────────────────────
def page_dashboard(df):
    st.markdown("## Tableau de bord financier")
    journaliser(st.session_state.username, "consultation", "Tableau de bord")

    charges  = df[df["type"] == "charge"]["montant"].sum()
    produits = df[df["type"] == "produit"]["montant"].sum()
    solde    = produits - charges
    nb_anomalies = df[df["anomalie"] == 1]["anomalie"].sum() if "anomalie" in df.columns else 0

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Produits",  f"{produits:,.0f} FCFA",  delta="2025")
    c2.metric("Total Charges",   f"{charges:,.0f} FCFA",   delta=None)
    c3.metric("Resultat Net",    f"{solde:,.0f} FCFA",
              delta="Benefice" if solde > 0 else "Deficit")
    c4.metric("Alertes Fraude",  f"{int(nb_anomalies)} transaction(s)",
              delta="A verifier" if nb_anomalies > 0 else None,
              delta_color="inverse")

    st.markdown("---")
    col_g, col_d = st.columns(2)

    # Évolution mensuelle
    with col_g:
        st.markdown("#### Evolution mensuelle")
        df_m = df.copy()
        df_m["mois"] = df_m["date"].dt.to_period("M").astype(str)
        mensuel = df_m.groupby(["mois", "type"])["montant"].sum().reset_index()
        fig = px.bar(
            mensuel, x="mois", y="montant", color="type",
            barmode="group",
            color_discrete_map={"produit": "#22c55e", "charge": "#ef4444"},
            labels={"montant": "FCFA", "mois": "", "type": ""},
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Répartition des charges
    with col_d:
        st.markdown("#### Repartition des charges")
        charges_df = df[df["type"] == "charge"].groupby("categorie")["montant"].sum().reset_index()
        charges_df["categorie"] = charges_df["categorie"].str.split(" - ").str[1]
        fig2 = px.pie(
            charges_df, values="montant", names="categorie",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig2.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Solde cumulé
    st.markdown("#### Evolution du solde cumulé")
    df_daily = df.groupby(["date", "type"])["montant"].sum().unstack(fill_value=0).reset_index()
    if "produit" not in df_daily: df_daily["produit"] = 0
    if "charge"  not in df_daily: df_daily["charge"]  = 0
    df_daily["solde_net"]    = df_daily["produit"] - df_daily["charge"]
    df_daily["solde_cumule"] = df_daily["solde_net"].cumsum()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_daily["date"], y=df_daily["solde_cumule"],
        fill="tozeroy", name="Solde cumulé",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59,130,246,0.1)",
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    fig3.update_layout(height=280, margin=dict(t=10, b=10),
                       yaxis_title="FCFA", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE 2 : CLASSIFICATION
# ────────────────────────────────────────────────────────────────────
def page_classification(clf):
    st.markdown("## Classification automatique des transactions")

    if not verifier_permission(st.session_state.role, "lire"):
        st.error("Acces refuse.")
        return

    tab1, tab2 = st.tabs(["Saisie manuelle", "Importer un CSV"])

    # ── Saisie manuelle ──────────────────────────────────────────────
    with tab1:
        st.markdown("Entrez le libellé d'une transaction pour la classifier automatiquement.")
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

    # ── Import CSV ───────────────────────────────────────────────────
    with tab2:
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
                        "Telecharger le résultat",
                        csv_out,
                        "transactions_classifiees.csv",
                        "text/csv",
                    )


# ────────────────────────────────────────────────────────────────────
#  PAGE 3 : BUDGET PRÉVISIONNEL
# ────────────────────────────────────────────────────────────────────
def page_budget():
    st.markdown("## Budget prévisionnel — Prophet (Meta)")
    journaliser(st.session_state.username, "consultation", "Budget previsionnel")

    horizon = st.slider("Horizon de prévision (jours)", 15, 90, 30, 15)

    with st.spinner("Calcul des prévisions..."):
        resultats = charger_previsions()
        resume    = resume_previsions(resultats, horizon_jours=horizon)
        mensuel, tendance = calculer_indicateurs(DATA_PATH)

    # KPIs prévisions
    c1, c2, c3 = st.columns(3)
    c1.metric("Produits prévus",
              f"{resume['produits']['total_prevu']:,.0f} FCFA",
              f"±{(resume['produits']['intervalle_haut'] - resume['produits']['intervalle_bas'])//2:,.0f}")
    c2.metric("Charges prévues",
              f"{resume['charges']['total_prevu']:,.0f} FCFA",
              f"±{(resume['charges']['intervalle_haut'] - resume['charges']['intervalle_bas'])//2:,.0f}")
    solde_prevu = resume['solde_net']['total_prevu']
    c3.metric("Solde prévu",
              f"{solde_prevu:,.0f} FCFA",
              "Excedent" if solde_prevu > 0 else "Deficit",
              delta_color="normal" if solde_prevu > 0 else "inverse")

    st.markdown("---")

    # Graphique prévisions vs historique
    col_g, col_d = st.columns(2)

    with col_g:
        st.markdown(f"#### Prévision des produits ({horizon}j)")
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
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("#### Résultats mensuels (6 derniers mois)")
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
            barmode="group", height=300, margin=dict(t=10, b=10),
            yaxis2=dict(overlaying="y", side="right"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"Tendance des 3 derniers mois : **{tendance.upper()}**")


# ────────────────────────────────────────────────────────────────────
#  PAGE 4 : ANOMALIES
# ────────────────────────────────────────────────────────────────────
def page_anomalies(df, iso):
    st.markdown("## Detection des transactions suspectes")
    journaliser(st.session_state.username, "consultation", "Detection anomalies")

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

    # Résumé
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analysées", len(df))
    c2.metric("Transactions suspectes", len(suspectes),
              delta=f"{len(suspectes)/len(df)*100:.1f}%", delta_color="inverse")
    c3.metric("Montant suspect total",
              f"{suspectes['montant'].sum():,.0f} FCFA")

    st.markdown("---")
    col_g, col_d = st.columns([3, 2])

    with col_g:
        st.markdown("#### Transactions suspectes")
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
            st.success("Aucune transaction suspecte détectée.")

    with col_d:
        st.markdown("#### Distribution des montants")
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
            margin=dict(t=10, b=10),
            xaxis_title="Montant (FCFA)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Alerte si anomalie flagrante
    critiques = suspectes[suspectes["score_anomalie"] < -0.04]
    if len(critiques) > 0:
        st.markdown("#### Alertes critiques")
        for _, row in critiques.iterrows():
            st.markdown(f"""
            <div class="alerte-rouge">
                <strong>ALERTE</strong> — {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}<br>
                {row['libelle']} — <strong>{row['montant']:,.0f} FCFA</strong><br>
                Score anomalie : {row['score_anomalie']:.4f}
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  PAGE 5 : JOURNAL D'AUDIT
# ────────────────────────────────────────────────────────────────────
def page_journal():
    st.markdown("## Journal d'audit — Chaine SHA-256")

    integre, anomalies = verifier_integrite_journal()

    if integre:
        st.markdown("""
        <div class="alerte-verte">
            <strong>INTEGRITE CONFIRMEE</strong> — La chaine de hachage est intacte.
            Aucune modification non autorisée détectée.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alerte-rouge">
            <strong>ALERTE — INTEGRITE COMPROMISE</strong>
        </div>
        """, unsafe_allow_html=True)
        for a in anomalies:
            st.error(a)

    st.markdown("---")
    entrees = lire_journal(100)

    if entrees:
        df_journal = pd.DataFrame(entrees)
        df_journal["hash"] = df_journal["hash"].apply(
            lambda h: f'<span class="hash-badge">{h}</span>'
        )

        # Filtres
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            users = ["Tous"] + list(df_journal["utilisateur"].unique())
            filtre_user = st.selectbox("Filtrer par utilisateur", users)
        with col_f2:
            actions = ["Toutes"] + list(df_journal["action"].unique())
            filtre_action = st.selectbox("Filtrer par action", actions)

        df_filtre = df_journal.copy()
        if filtre_user   != "Tous":    df_filtre = df_filtre[df_filtre["utilisateur"] == filtre_user]
        if filtre_action != "Toutes":  df_filtre = df_filtre[df_filtre["action"]      == filtre_action]

        st.markdown(f"**{len(df_filtre)} entrées**")
        st.write(
            df_filtre[["id", "horodatage", "utilisateur", "action", "details", "hash"]]
            .to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.info("Le journal est vide.")


# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.connecte:
        page_login()
        return

    clf, iso = charger_modeles()
    df       = charger_donnees()
    page     = sidebar()

    if page == "Tableau de bord":
        page_dashboard(df)
    elif page == "Classification":
        page_classification(clf)
    elif page == "Budget previsionnel":
        page_budget()
    elif page == "Anomalies":
        page_anomalies(df, iso)
    elif page == "Journal d'audit":
        page_journal()


if __name__ == "__main__":
    main()
