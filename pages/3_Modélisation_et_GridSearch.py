import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, log_loss, classification_report
)
from scipy import sparse

# Modèles
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="Modèles & GridSearch", page_icon="🤖", layout="wide")
st.title("🤖 Modélisation & Optimisation Avancée")

# CSS personnalisé avec animations
st.markdown("""
<style>
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: rgba(255,255,255,0.3);
    }
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .metric-box {
        background: rgba(255,255,255,0.15);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        background: rgba(255,255,255,0.25);
        transform: scale(1.05);
    }
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .param-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .param-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .success-animation {
        animation: successPulse 1.5s ease-in-out;
    }
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: progressAnimation 2s ease-in-out infinite;
    }
    @keyframes progressAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .model-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 0) Chargement des données avec animations
# =========================================================
st.markdown('<div class="fade-in">', unsafe_allow_html=True)

use_session_preprocessed = all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"])

if use_session_preprocessed:
    st.success("✅ Données prétraitées chargées depuis la session")
    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = np.array(st.session_state["y_train"])
    y_test = np.array(st.session_state["y_test"])

    # Récupération des paramètres GridSearch si disponibles
    gridsearch_params = st.session_state.get("gridsearch_params", {
        "cv_folds": 5,
        "scoring_metric": "roc_auc",
        "n_jobs": -1,
        "verbosity": 0,
        "refit": True
    })

    # Affichage des métriques de base avec animations
    st.subheader("📊 Aperçu du Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Train samples", f"{X_train.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Test samples", f"{X_test.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Features", X_train.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        pos_rate = y_test.mean()
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Taux positif (test)", f"{pos_rate:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Fallback vers df_ready ou df_raw
    if "df_ready" in st.session_state:
        df = st.session_state["df_ready"].copy()
        st.info("Utilisation de **df_ready** (dataset prêt numérique).")
    elif "df_raw" in st.session_state:
        df = st.session_state["df_raw"].copy()
        st.warning("Utilisation de **df_raw** - assurez-vous qu'il est 100% numérique")
    else:
        st.error("❌ Aucun dataset disponible. Retournez à la page **Upload**.")
        st.stop()

    st.subheader("📌 Configuration finale")
    target_col = st.selectbox(
        "Colonne cible",
        options=df.columns.tolist(),
        index=(df.columns.tolist().index("readmitted_num") if "readmitted_num" in df.columns else 0)
    )

    if df[target_col].nunique() != 2:
        st.error("La cible doit être binaire (0/1)")
        st.stop()

    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    # Vérification données catégorielles
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        st.error(f"❌ Colonnes catégorielles détectées: {cat_cols}")
        st.stop()

    test_size = st.slider("Taille du test set", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    gridsearch_params = {
        "cv_folds": 5,
        "scoring_metric": "roc_auc",
        "n_jobs": -1,
        "verbosity": 0,
        "refit": True
    }

st.markdown('</div>', unsafe_allow_html=True)


def to_dense_if_needed(A):
    return A.toarray() if sparse.issparse(A) else A


# =========================================================
# 1) Sélection des modèles avec interface améliorée
# =========================================================
st.markdown("---")
st.subheader("🧠 Sélection des Algorithmes")

# Section de sélection des modèles avec design amélioré
st.markdown('<div class="slide-in">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    m_nb = st.checkbox("🧪 Naive Bayes (GaussianNB)", value=True)
    m_log = st.checkbox("📈 Régression Logistique", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    m_svm = st.checkbox("⚡ SVM Linéaire", value=True)
    m_dt = st.checkbox("🌳 Arbre de Décision", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    m_rf = st.checkbox("🌲 Random Forest", value=True)
    m_xgb = st.checkbox("🚀 XGBoost", value=HAS_XGB, disabled=not HAS_XGB)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Alertes modèles lourds avec animation
selected_models = []
if m_nb: selected_models.append("NaiveBayes")
if m_log: selected_models.append("LogReg")
if m_svm: selected_models.append("SVM")
if m_dt: selected_models.append("DecisionTree")
if m_rf: selected_models.append("RandomForest")
if m_xgb: selected_models.append("XGBoost")

if selected_models:
    st.markdown(f"""
    <div class="fade-in" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
        <h4 style="margin:0; display: flex; align-items: center; gap: 0.5rem;">
            📋 Modèles sélectionnés ({len(selected_models)})
        </h4>
        <div style="margin-top: 0.5rem;">
            {''.join([f'<span class="model-badge">{model}</span>' for model in selected_models])}
        </div>
    </div>
    """, unsafe_allow_html=True)

heavy_models = []
if m_rf: heavy_models.append("Random Forest")
if m_xgb: heavy_models.append("XGBoost")
if heavy_models:
    st.warning(f"⏱️ **Modèles potentiellement lents** : {', '.join(heavy_models)} - Peut prendre plusieurs minutes")

# =========================================================
# 2) Configuration avancée du GridSearch
# =========================================================
st.markdown("---")
st.subheader("⚙️ Configuration Avancée du GridSearch")

with st.expander("🎛️ Paramètres de Recherche Hyperparamètres", expanded=True):
    st.markdown('<div class="param-section">', unsafe_allow_html=True)

    gs_col1, gs_col2 = st.columns(2)

    with gs_col1:
        cv_folds = st.slider("**Nombre de folds CV**", 3, 10, gridsearch_params["cv_folds"],
                             help="Nombre de folds pour la validation croisée")
        scoring_metric = st.selectbox("**Métrique d'évaluation**",
                                      ["roc_auc", "accuracy", "f1", "precision", "recall"],
                                      index=["roc_auc", "accuracy", "f1", "precision", "recall"].index(
                                          gridsearch_params["scoring_metric"]),
                                      help="Métrique utilisée pour évaluer les modèles")

    with gs_col2:
        n_jobs = st.selectbox("**Parallélisation**",
                              [1, 2, 4, -1],
                              index=[1, 2, 4, -1].index(gridsearch_params["n_jobs"]),
                              format_func=lambda x: f"{x} core(s)" if x != -1 else "Tous les cores (-1)",
                              help="Nombre de jobs parallèles")
        refit = st.checkbox("**Refit automatique**", value=gridsearch_params["refit"],
                            help="Re-entraîne le meilleur modèle sur toutes les données d'entraînement")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 3) Configuration des grilles de paramètres
# =========================================================
grids = {}

if m_nb:
    grids["NaiveBayes"] = {
        "prep": lambda Xt, Xv: (to_dense_if_needed(Xt), to_dense_if_needed(Xv)),
        "estimator": GaussianNB(),
        "params": {},
        "description": "🧪 Classifieur probabiliste basé sur le théorème de Bayes"
    }

if m_log:
    grids["LogReg"] = {
        "prep": lambda Xt, Xv: (Xt, Xv),
        "estimator": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"]
        },
        "description": "📈 Modèle linéaire avec régularisation L1/L2"
    }

if m_svm:
    grids["SVM_linear"] = {
        "prep": lambda Xt, Xv: (Xt, Xv),
        "estimator": SVC(kernel="linear", probability=True, random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0]
        },
        "description": "⚡ Machine à Vecteurs de Support avec noyau linéaire"
    }

if m_dt:
    grids["DecisionTree"] = {
        "prep": lambda Xt, Xv: (Xt, Xv),
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "description": "🌳 Arbre de décision avec contrôle de la profondeur"
    }

if m_rf:
    grids["RandomForest"] = {
        "prep": lambda Xt, Xv: (Xt, Xv),
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        "description": "🌲 Ensemble d'arbres de décision (Bagging)"
    }

if m_xgb and HAS_XGB:
    grids["XGBoost"] = {
        "prep": lambda Xt, Xv: (Xt, Xv),
        "estimator": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        "params": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        },
        "description": "🚀 Gradient Boosting optimisé et très performant"
    }

# =========================================================
# 4) Interface de lancement avec animations
# =========================================================
st.markdown("---")
st.subheader("🚀 Lancement de l'Entraînement")

if not grids:
    st.warning("⚠️ Veuillez sélectionner au moins un modèle")
    st.stop()

# Résumé de la configuration
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; margin: 1rem 0;">
    <h4 style="margin:0; display: flex; align-items: center; gap: 0.5rem;">🎯 Configuration Résumée</h4>
    <div style="margin-top: 1rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
        <div><strong>Modèles sélectionnés:</strong><br>{}</div>
        <div><strong>CV Folds:</strong><br>{}</div>
        <div><strong>Métrique:</strong><br>{}</div>
        <div><strong>Parallélisation:</strong><br>{}</div>
    </div>
</div>
""".format(len(grids), cv_folds, scoring_metric, f"{n_jobs} core(s)" if n_jobs != -1 else "Tous les cores"),
            unsafe_allow_html=True)

if st.button("🎯 Lancer l'Entraînement et GridSearch", type="primary", use_container_width=True, key="train_button"):

    # Animation de lancement
    with st.spinner("🚀 Initialisation de l'entraînement..."):
        time.sleep(1)

    # Barre de progression principale
    main_progress = st.progress(0)
    status_text = st.empty()

    results = []
    trained_models = {}
    best_params_all = {}
    all_curves_data = {}

    # Étape 1: Initialisation
    status_text.text("🔧 Préparation des données et initialisation des modèles...")
    main_progress.progress(10)
    time.sleep(0.5)

    for i, (name, spec) in enumerate(grids.items()):
        # Mise à jour du statut avec animation
        status_text.text(f"🎯 Entraînement de {name}... ({i + 1}/{len(grids)})")

        # Barre de progression pour le modèle courant
        model_progress = st.progress(0)

        # Préparation données
        Xt, Xv = spec["prep"](X_train, X_test)
        model_progress.progress(25)

        # GridSearch avec paramètres configurés
        gs = GridSearchCV(
            spec["estimator"],
            param_grid=spec["params"],
            cv=cv_folds,
            scoring=scoring_metric,
            n_jobs=n_jobs,
            verbose=0,
            refit=refit
        )

        model_progress.progress(50)
        gs.fit(Xt, y_train)
        model_progress.progress(75)

        best_model = gs.best_estimator_
        trained_models[name] = best_model
        best_params_all[name] = gs.best_params_

        # Prédictions
        y_pred = best_model.predict(Xv)
        y_proba = best_model.predict_proba(Xv)[:, 1] if hasattr(best_model, "predict_proba") else None

        # Métriques complètes
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
            "PR_AUC": average_precision_score(y_test, y_proba) if y_proba is not None else np.nan,
            "LogLoss": log_loss(y_test, y_proba) if y_proba is not None else np.nan,
        }

        # Stockage courbes
        if y_proba is not None:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)

            all_curves_data[name] = {
                "fpr": fpr, "tpr": tpr,
                "precision": precision, "recall": recall,
                "roc_auc": metrics["ROC_AUC"],
                "pr_auc": metrics["PR_AUC"]
            }

        results.append(metrics)
        model_progress.progress(100)
        main_progress.progress(10 + (i + 1) * (80 // len(grids)))

        # Petite pause pour l'animation
        time.sleep(0.3)

    # Finalisation
    status_text.text("📊 Génération des résultats et visualisations...")
    main_progress.progress(95)
    time.sleep(1)

    status_text.text("✅ Entraînement terminé avec succès!")
    main_progress.progress(100)
    time.sleep(0.5)

    # Animation de célébration
    st.balloons()

    # =========================================================
    # 5) Affichage des résultats avec animations
    # =========================================================

    st.markdown("---")
    st.subheader("🏆 Résultats de l'Entraînement")

    # DataFrame des résultats
    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

    # Classement avec animation
    st.markdown('<div class="success-animation">', unsafe_allow_html=True)

    # Meilleur modèle highlight
    best_model_name = results_df.iloc[0]["Model"]
    best_roc_auc = results_df.iloc[0]["ROC_AUC"]

    st.success(f"""
    🎉 **Meilleur modèle : {best_model_name}** 

    **Score {scoring_metric.upper()} : {best_roc_auc:.3f}**
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Métriques formatées avec style
    st.markdown("### 📈 Tableau Comparatif des Performances")

    styled_df = results_df.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1": "{:.3f}", "ROC_AUC": "{:.3f}", "PR_AUC": "{:.3f}", "LogLoss": "{:.3f}"
    }).background_gradient(subset=["ROC_AUC", "F1"], cmap="Blues")

    st.dataframe(styled_df, use_container_width=True, height=400)

    # =========================================================
    # 6) Visualisations détaillées avec Plotly
    # =========================================================

    st.markdown("---")
    st.subheader("📊 Visualisations des Performances")

    # Courbes ROC avec animation
    if all_curves_data:
        st.markdown("### 📈 Courbes ROC Comparées")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(dash='dash', color='gray'),
                                     name='Aléatoire', showlegend=False))

        for name, curves in all_curves_data.items():
            fig_roc.add_trace(go.Scatter(
                x=curves["fpr"], y=curves["tpr"],
                mode='lines',
                name=f'{name} (AUC = {curves["roc_auc"]:.3f})',
                line=dict(width=3)
            ))

        fig_roc.update_layout(
            title="Courbes ROC - Comparaison des modèles",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=800, height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Courbes Precision-Recall
        st.markdown("### 📊 Courbes Precision-Recall")
        fig_pr = go.Figure()

        for name, curves in all_curves_data.items():
            fig_pr.add_trace(go.Scatter(
                x=curves["recall"], y=curves["precision"],
                mode='lines',
                name=f'{name} (AP = {curves["pr_auc"]:.3f})',
                line=dict(width=3)
            ))

        fig_pr.update_layout(
            title="Courbes Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=800, height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Matrices de confusion pour les 3 meilleurs modèles
    st.markdown("### 🎯 Matrices de Confusion")
    top_models = results_df.head(3)["Model"].tolist()

    conf_cols = st.columns(3)
    for idx, model_name in enumerate(top_models):
        if model_name in trained_models:
            with conf_cols[idx]:
                model = trained_models[model_name]
                Xt, Xv = grids[model_name]["prep"](X_train, X_test)
                y_pred = model.predict(Xv)
                cm = confusion_matrix(y_test, y_pred)

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title=f"{model_name}",
                    aspect="auto"
                )
                fig_cm.update_layout(
                    xaxis_title="Prédit",
                    yaxis_title="Réel",
                    width=350,
                    height=350
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # =========================================================
    # 7) Hyperparamètres et sauvegarde
    # =========================================================

    st.markdown("---")
    st.subheader("⚙️ Hyperparamètres Optimaux")

    # Affichage des hyperparamètres dans des expanders
    for name, params in best_params_all.items():
        with st.expander(f"🔧 {name} - Paramètres optimaux", expanded=False):
            st.json(params)

    # Sauvegarde avec animation
    st.markdown("### 💾 Sauvegarde du Modèle")

    save_col1, save_col2 = st.columns([2, 1])

    with save_col1:
        if st.button("💾 Sauvegarder le meilleur modèle", use_container_width=True, type="primary"):
            try:
                import joblib

                best_model = trained_models[best_model_name]
                filename = f"best_model_{best_model_name}.pkl"
                joblib.dump(best_model, filename)

                st.success(f"""
                ✅ **Modèle sauvegardé avec succès!**

                **Fichier :** `{filename}`
                **Modèle :** {best_model_name}
                **Score :** {best_roc_auc:.3f}
                """)

                # Animation de confirmation
                st.balloons()

            except Exception as e:
                st.error(f"❌ Erreur lors de la sauvegarde: {e}")

    with save_col2:
        if st.button("🔄 Exporter tous les résultats", use_container_width=True):
            # Export des résultats complets
            results_df.to_csv("model_comparison_results.csv", index=False)
            st.info("📄 Résultats exportés dans `model_comparison_results.csv`")

else:
    st.info("""
    👆 **Prêt à lancer l'entraînement?** 

    Cliquez sur le bouton ci-dessus pour démarrer le GridSearch sur les modèles sélectionnés.
    Le processus peut prendre plusieurs minutes selon le nombre de modèles et la taille des données.
    """)

# =========================================================
# 8) Footer informatif
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 <strong>Module de Modélisation Avancée</strong> - GridSearch avec validation croisée</p>
    <p style="font-size: 0.9rem;">Optimisation hyperparamètres • Validation croisée • Métriques multiples</p>
</div>
""", unsafe_allow_html=True)