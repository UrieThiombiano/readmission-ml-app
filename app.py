import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Readmission ML App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .feature-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4; margin-bottom: 1rem; }
    .info-box { background-color: #e8f4fd; padding: 1rem; border-radius: 8px; border: 1px solid #bee5eb; }
    .step-container { display: flex; justify-content: space-between; margin: 2rem 0; }
    .step { text-align: center; flex: 1; padding: 1rem; }
    .step-number { background-color: #1f77b4; color: white; border-radius: 50%; width: 40px; height: 40px;
                   display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🩺 Application ML de Prédiction de Réadmission</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>Bienvenue dans votre outil d'analyse prédictive des réadmissions hospitalières</b><br>
Cette application vous guide à travers un processus complet : upload → prétraitement → modélisation → comparaison.
</div>
""", unsafe_allow_html=True)

# Étapes
st.subheader("🚀 Processus en 4 étapes")
cols = st.columns(4)
labels = [("1", "📥 Upload des données", "Importez un CSV/Excel"),
          ("2", "🔍 Analyse exploratoire", "Aperçu & stats"),
          ("3", "⚙️ Prétraitement", "Split / OHE / scaling"),
          ("4", "🤖 Modélisation", "Entraînement & comparaison")]
for i, (num, title, desc) in enumerate(labels):
    with cols[i]:
        st.markdown(f"""
        <div class="step">
            <div class="step-number">{num}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Fonctionnalités
st.subheader("🎯 Fonctionnalités principales")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Analyse des données</h4>
        <ul><li>Statistiques descriptives</li><li>Visualisations</li><li>Manquants</li><li>Corrélations</li></ul>
    </div>
    <div class="feature-card">
        <h4>⚙️ Prétraitement avancé</h4>
        <ul><li>Encodage catégorielles</li><li>Normalisation</li><li>Gestion déséquilibre</li></ul>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 Algorithmes</h4>
        <ul><li>Régression Logistique</li><li>Random Forest</li><li>XGBoost</li><li>Comparaison</li></ul>
    </div>
    <div class="feature-card">
        <h4>📈 Évaluation</h4>
        <ul><li>AUC / F1 / Recall / Precision</li><li>ROC / PR</li><li>Matrice de confusion</li></ul>
    </div>
    """, unsafe_allow_html=True)

st.subheader("🎯 Démarrer")
# Lien natif (Streamlit ≥ 1.31), sinon fallback bouton + switch_page
try:
    st.page_link("pages/1_upload_preview.py", label="📥 Aller à : Upload et Aperçu", icon="📥")
except Exception:
    if st.button("📥 Aller à : Upload et Aperçu"):
        st.switch_page("pages/1_Upload_et_Aperçu.py")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Application développée avec Streamlit • Optimisée pour l'analyse des données de santé</div>",
    unsafe_allow_html=True
)
