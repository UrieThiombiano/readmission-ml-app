import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from scipy import sparse
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Préprocessing", page_icon="⚙️", layout="wide")
st.title("⚙️ Préprocessing — Préparation des Données")

# CSS personnalisé avec animations
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .success-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .navigation-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
    }
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .param-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .param-section:hover {
        background: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# --- 0) Vérifications initiales ---
if "df_raw" not in st.session_state:
    st.error("📭 Aucun dataset en mémoire. Retournez à la page **Upload**.")
    st.stop()

df = st.session_state["df_raw"].copy()

# Header informatif avec animation
st.markdown("""
<div class="fade-in" style="background-color: #e8f4fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <h3 style="margin:0; color: #155724;">🎯 Objectif du Préprocessing</h3>
    <p style="margin:0.5rem 0 0 0;">Transformer votre dataset en format <strong>100% numérique</strong> prêt pour l'entraînement ML</p>
</div>
""", unsafe_allow_html=True)

# --- 1) Analyse exploratoire initiale ---
st.subheader("📊 Analyse Exploratoire Initiale")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Lignes", f"{df.shape[0]:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Colonnes", df.shape[1])
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    missing_total = df.isna().sum().sum()
    missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Données manquantes", f"{missing_pct:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    numeric_cols = df.select_dtypes(include=['number']).shape[1]
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Colonnes numériques", numeric_cols)
    st.markdown('</div>', unsafe_allow_html=True)

# Visualisations rapides avec tabs animés
tab1, tab2, tab3 = st.tabs(["📋 Aperçu données", "📈 Types de données", "⚠️ Données manquantes"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True, height=300)

with tab2:
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig_dtype = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                       title="Répartition des types de données")
    st.plotly_chart(fig_dtype, use_container_width=True)

with tab3:
    missing_data = df.isna().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    if len(missing_data) > 0:
        fig_missing = px.bar(x=missing_data.values, y=missing_data.index, orientation='h',
                             title="Données manquantes par colonne",
                             labels={'x': 'Nombre de valeurs manquantes', 'y': 'Colonnes'})
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ Aucune donnée manquante détectée")

# --- 2) Configuration du préprocessing ---
st.subheader("⚙️ Configuration du Préprocessing")

# Sélection de la cible
cols = df.columns.tolist()
target = st.selectbox(
    "**Colonne cible** (variable à prédire)",
    options=cols,
    index=(cols.index("readmitted_num") if "readmitted_num" in cols else
           cols.index("readmission") if "readmission" in cols else
           cols.index("target") if "target" in cols else 0),
    help="Sélectionnez la variable que vous souhaitez prédire"
)

# Affichage info cible
if target:
    target_info = df[target].value_counts()
    st.write(f"**Distribution de la cible `{target}`:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Valeurs uniques", target_info.shape[0])
    with col2:
        st.metric("Valeur majoritaire", f"{target_info.index[0]} ({target_info.iloc[0] / len(df) * 100:.1f}%)")
    with col3:
        if target_info.shape[0] == 2:
            imbalance_ratio = min(target_info) / max(target_info)
            st.metric("Ratio déséquilibre", f"{imbalance_ratio:.2f}")

# Options de prétraitement
st.markdown("**Options de prétraitement:**")
c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
    drop_dups = st.checkbox("Supprimer les doublons", value=True)
with c2:
    random_state = st.number_input("Random state", 0, 9999, 42)
    scale_numeric = st.checkbox("Standardiser les numériques", value=True)
with c3:
    min_freq = st.number_input("Fréquence min. catégories", min_value=0.0, max_value=0.2, value=0.01, step=0.01,
                               help="Regroupe les catégories rares")
    handle_unknown = st.selectbox("Gestion nouvelles catégories",
                                  options=["infrequent_if_exist", "ignore"], index=0)

# Gestion des doublons
if drop_dups:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        st.info(f"🗑️ Doublons supprimés : {before - after} ligne(s)")

# --- 3) Sélection des caractéristiques ---
st.subheader("🎯 Sélection des Caractéristiques")

y = df[target]
X = df.drop(columns=[target])

# Détection automatique
detected_num = X.select_dtypes(include=np.number).columns.tolist()
detected_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Colonnes Numériques**")
    num_cols = st.multiselect(
        "Sélectionnez les colonnes numériques",
        options=X.columns.tolist(),
        default=detected_num,
        help="Variables continues ou discrètes"
    )

with col2:
    st.markdown("**Colonnes Catégorielles**")
    cat_cols = st.multiselect(
        "Sélectionnez les colonnes catégorielles",
        options=[c for c in X.columns if c not in num_cols],
        default=detected_cat,
        help="Variables qualitatives (seront encodées)"
    )

# Colonnes ignorées
missing = [c for c in X.columns if c not in num_cols + cat_cols]
if missing:
    st.warning(f"⚠️ Colonnes ignorées (ni numériques ni catégorielles) : {', '.join(missing)}")

# --- 4) Pipelines de prétraitement ---
st.subheader("🔧 Pipelines de Transformation")

# Pipelines
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    *([("scaler", StandardScaler())] if scale_numeric else [])
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=True,
        min_frequency=min_freq if handle_unknown == "infrequent_if_exist" else None
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    remainder="drop"
)

# --- 5) Configuration du GridSearch ---
st.subheader("🎛️ Configuration du GridSearch")

with st.expander("⚙️ Paramètres avancés du GridSearch", expanded=True):
    st.markdown("""
    <div class="param-section">
    <h4>🔧 Paramètres de recherche</h4>
    """, unsafe_allow_html=True)

    gs_col1, gs_col2, gs_col3 = st.columns(3)

    with gs_col1:
        cv_folds = st.slider("Nombre de folds CV", 3, 10, 5,
                             help="Nombre de folds pour la validation croisée")
        scoring_metric = st.selectbox("Métrique d'évaluation",
                                      ["roc_auc", "accuracy", "f1", "precision", "recall"],
                                      index=0,
                                      help="Métrique utilisée pour évaluer les modèles")

    with gs_col2:
        n_jobs = st.selectbox("Parallélisation",
                              [1, 2, 4, -1],
                              index=3,
                              format_func=lambda x: f"{x} core(s)" if x != -1 else "Tous les cores",
                              help="Nombre de jobs parallèles (-1 = tous les cores)")
        verbosity = st.selectbox("Verbosite",
                                 [0, 1, 2, 3],
                                 index=0,
                                 help="Niveau de détail des logs (0 = silencieux)")

    with gs_col3:
        refit = st.checkbox("Refit automatique", value=True,
                            help="Re-entraîne le meilleur modèle sur toutes les données")
        enable_cache = st.checkbox("Cache GridSearch", value=True,
                                   help="Active le cache pour accélérer les recherches")

    st.markdown("</div>", unsafe_allow_html=True)

# --- 6) Application du préprocessing ---
st.subheader("🚀 Application des Transformations")

if st.button("🔨 Appliquer le Préprocessing", type="primary", use_container_width=True):

    # Barre de progression avec animation
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        "Initialisation...",
        "Split des données...",
        "Préprocessing numérique...",
        "Encodage catégoriel...",
        "Finalisation..."
    ]

    for i, step in enumerate(steps):
        status_text.text(f"🔄 {step}")
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.3)  # Animation fluide

    with st.spinner("Application des transformations..."):
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if y.nunique() == 2 else None
        )

        # Fit et transform
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # Feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = np.array([f"feature_{i}" for i in range(X_train_t.shape[1])])

        # Sauvegarde en session avec paramètres GridSearch
        st.session_state.update({
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "X_train": X_train_t,
            "X_test": X_test_t,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessing_applied": True,
            "gridsearch_params": {
                "cv_folds": cv_folds,
                "scoring_metric": scoring_metric,
                "n_jobs": n_jobs,
                "verbosity": verbosity,
                "refit": refit,
                "enable_cache": enable_cache
            }
        })

    # Affichage des résultats avec animation
    status_text.text("✅ Préprocessing terminé !")
    progress_bar.empty()

    st.balloons()  # Animation de célébration

    # Métriques finales
    st.markdown("### 📊 Métriques Finales du Préprocessing")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features finales", X_train_t.shape[1])
    with col2:
        st.metric("Train size", f"{X_train_t.shape[0]:,}")
    with col3:
        st.metric("Test size", f"{X_test_t.shape[0]:,}")
    with col4:
        if sparse.issparse(X_train_t):
            density = (X_train_t.nnz / (X_train_t.shape[0] * X_train_t.shape[1])) * 100
        else:
            non_zero_count = np.count_nonzero(X_train_t)
            density = (non_zero_count / (X_train_t.shape[0] * X_train_t.shape[1])) * 100
        st.metric("Densité matrice", f"{density:.1f}%")

    # Aperçu des features
    with st.expander("👁️ Aperçu des Features Transformées", expanded=False):
        try:
            if sparse.issparse(X_train_t):
                X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_t, columns=feature_names)
            else:
                X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
            st.dataframe(X_train_df.iloc[:8, :12], use_container_width=True)

            st.write(f"**Type de matrice :** {'Sparse' if sparse.issparse(X_train_t) else 'Dense'}")
            st.write(f"**Format :** {X_train_t.shape[0]} × {X_train_t.shape[1]}")

        except Exception as e:
            st.error(f"Erreur lors de la création de l'aperçu : {e}")
            st.info("Aperçu non disponible pour les grandes matrices")

    # Informations sur l'encodage
    with st.expander("🔍 Détails de l'Encodage", expanded=False):
        try:
            num_features = len(num_cols)
            cat_features = 0
            for col in cat_cols:
                try:
                    unique_vals = X[col].nunique()
                    cat_features += unique_vals
                except:
                    cat_features += 1

            st.write(f"**Features numériques originales :** {num_features}")
            st.write(f"**Features catégorielles originales :** {len(cat_cols)}")
            st.write(f"**Features totales après encodage :** {len(feature_names)}")
            st.write(f"**Facteur d'expansion :** {len(feature_names) / (num_features + len(cat_cols)):.1f}x")

        except Exception as e:
            st.write("Informations d'encodage non disponibles")

    # Sauvegarde optionnelle
    st.markdown("### 💾 Sauvegarde")
    if st.button("💾 Sauvegarder les Artéfacts", use_container_width=True):
        with st.spinner("Sauvegarde en cours..."):
            Path("models").mkdir(exist_ok=True)
            joblib.dump(preprocessor, "models/preprocessor.pkl")
            joblib.dump({"feature_names": feature_names}, "models/meta.pkl")

            # Sauvegarde des paramètres GridSearch
            gridsearch_config = {
                "cv_folds": cv_folds,
                "scoring_metric": scoring_metric,
                "n_jobs": n_jobs,
                "verbosity": verbosity,
                "refit": refit
            }
            joblib.dump(gridsearch_config, "models/gridsearch_config.pkl")

            try:
                if sparse.issparse(X_train_t):
                    sparse.save_npz("models/X_train.npz", X_train_t)
                    sparse.save_npz("models/X_test.npz", X_test_t)
                else:
                    np.save("models/X_train.npy", X_train_t)
                    np.save("models/X_test.npy", X_test_t)

                pd.Series(y_train).to_csv("models/y_train.csv", index=False)
                pd.Series(y_test).to_csv("models/y_test.csv", index=False)
                st.success("✅ Données transformées sauvegardées dans `models/`")
            except Exception as e:
                st.warning(f"Données transformées non sauvegardées : {e}")

            st.success("Artéfacts sauvegardés dans `models/`")

# --- 7) Navigation vers la page suivante ---
st.markdown("---")
st.subheader("🎯 Navigation")

# Vérifier si le préprocessing a été fait dans cette session
preprocessing_applied = st.session_state.get("preprocessing_applied", False)

if preprocessing_applied:
    st.success("✅ Préprocessing terminé ! Vous pouvez maintenant passer à la modélisation.")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🚀 Passer à la Modélisation", type="primary", use_container_width=True):
            try:
                # Animation de transition
                with st.spinner("Chargement de la page de modélisation..."):
                    time.sleep(1)
                    st.switch_page("pages/3_Modélisation_et_GridSearch.py")
            except Exception as e:
                st.error(f"Erreur de navigation : {e}")

    with col2:
        if st.button("🔄 Recommencer le Préprocessing", type="secondary", use_container_width=True):
            # Réinitialiser les variables de session
            keys_to_remove = ["preprocessor", "feature_names", "X_train", "X_test", "y_train", "y_test",
                              "preprocessing_applied", "gridsearch_params"]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

else:
    st.info(
        "👆 Cliquez sur **'Appliquer le Préprocessing'** pour préparer vos données avant de passer à la modélisation.")

# --- 8) Instructions supplémentaires ---
with st.expander("📋 Instructions importantes", expanded=False):
    st.markdown("""
    **Pour une modélisation réussie :**

    1. **Sélectionnez la bonne variable cible** - doit être binaire (0/1) pour la classification
    2. **Vérifiez les types de colonnes** - numériques vs catégorielles
    3. **Configurez le GridSearch** - paramètres de recherche avancés
    4. **Appliquez le préprocessing** - cliquez sur le bouton bleu ci-dessus
    5. **Passez à la modélisation** - une fois le préprocessing terminé

    **Conseils GridSearch :**
    - **CV Folds** : 5-10 folds pour un bon compromis performance/stabilité
    - **Métrique** : ROC-AUC recommandée pour les problèmes déséquilibrés
    - **Parallélisation** : Utilisez tous les cores (-1) pour accélérer
    - **Cache** : Activez le cache pour les recherches répétées
    """)