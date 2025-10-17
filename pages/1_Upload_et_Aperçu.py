import io
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Upload Dataset - Readmission ML", page_icon="📥", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .upload-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 0.5rem 0; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .file-info { background-color: #e8f4fd; padding: 1rem; border-radius: 8px; border: 1px solid #bee5eb; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="upload-header">
    <h1 style="margin:0; color:white;">📥 Upload et Exploration du Dataset</h1>
    <p style="margin:0; opacity:0.9;">Chargez votre fichier de données pour commencer</p>
</div>
""", unsafe_allow_html=True)

MAX_MB = 50
SUPPORTED_TYPES = ["csv", "xlsx", "xls"]

@st.cache_data(show_spinner="Analyse du fichier...")
def read_csv_safely(file_bytes: bytes, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            pass
        # Sniff sep + encodage fallback
        sep = ","
        sample = file_bytes[:20000].decode("utf-8", errors="ignore")
        try:
            sep = pd.io.common.csv.Sniffer().sniff(sample).delimiter
        except Exception:
            pass
        for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=enc)
            except Exception:
                continue
        raise ValueError("Impossible de lire le CSV (encodage/séparateur).")
    else:
        # Excel
        return pd.read_excel(io.BytesIO(file_bytes))

def display_data_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><h4 style="margin:0;color:#667eea;">📊 Lignes</h4><h2 style="margin:0;">{df.shape[0]:,}</h2></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h4 style="margin:0;color:#667eea;">📈 Colonnes</h4><h2 style="margin:0;">{df.shape[1]}</h2></div>""", unsafe_allow_html=True)
    with col3:
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.size else 0.0
        st.markdown(f"""<div class="metric-card"><h4 style="margin:0;color:#667eea;">⚠️ Manquants</h4><h2 style="margin:0;">{missing_pct:.1f}%</h2></div>""", unsafe_allow_html=True)
    with col4:
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        st.markdown(f"""<div class="metric-card"><h4 style="margin:0;color:#667eea;">🔢 Numériques</h4><h2 style="margin:0;">{numeric_cols}</h2></div>""", unsafe_allow_html=True)

# --- Upload ---
st.subheader("📤 Chargement du fichier")
c_up, c_info = st.columns([2, 1])
with c_up:
    uploaded_file = st.file_uploader(
        "Déposez un fichier CSV ou Excel",
        type=SUPPORTED_TYPES,
        help=f"Formats: {', '.join(SUPPORTED_TYPES).upper()} • Taille max: {MAX_MB}MB"
    )
with c_info:
    st.markdown(f"""
    <div class="file-info">
        <h4>📋 Formats supportés</h4>
        <ul style="margin:0; padding-left:1rem;">
            <li>CSV (recommandé)</li><li>Excel (XLSX, XLS)</li>
        </ul>
        <p><strong>Taille max:</strong> {MAX_MB}MB</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="warning-box">
        <h4>👋 Prêt à commencer ?</h4>
        <p>Déposez un fichier CSV/Excel pour explorer vos données.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("💡 Structure recommandée"):
        st.markdown("""
        - 1 ligne = 1 observation • colonnes = features • cible binaire (0/1) si classification  
        - En-têtes clairs, séparateur `,`, encodage UTF-8 conseillé.
        """)
    st.stop()

# --- Lecture ---
size_mb = uploaded_file.size / (1024**2)
if size_mb > MAX_MB:
    st.error(f"❌ Fichier trop volumineux ({size_mb:.1f} Mo > {MAX_MB} Mo)")
    st.stop()

try:
    file_type = "csv" if uploaded_file.name.lower().endswith(".csv") else "excel"
    df = read_csv_safely(uploaded_file.getvalue(), file_type)
except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()

# --- Session ---
st.session_state["df_raw"] = df
st.session_state["file_info"] = {
    "name": uploaded_file.name,
    "size_mb": size_mb,
    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "shape": df.shape
}

st.markdown(f"""
<div class="success-box">
    <h4 style="margin:0; color:#155724;">✅ Fichier chargé</h4>
    <p style="margin:0;"><b>Nom :</b> {uploaded_file.name} • <b>Taille :</b> {size_mb:.1f} MB</p>
    <p style="margin:0;"><b>Dimensions :</b> {df.shape[0]:,} lignes × {df.shape[1]} colonnes</p>
</div>
""", unsafe_allow_html=True)

# --- Aperçu & métriques ---
st.subheader("📊 Métriques")
display_data_metrics(df)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Données", "📈 Types", "⚠️ Manquants", "📏 Stats"])
with tab1:
    n = st.slider("Nombre de lignes à afficher", 5, 100, 10)
    st.dataframe(df.head(n), use_container_width=True, height=380)
with tab2:
    st.write("**Dtypes** :", df.dtypes.astype(str).to_dict())
    st.write("**Uniques (top)** :")
    for col in df.columns[:15]:
        st.write(f"- `{col}` : {df[col].nunique()} valeurs uniques")
with tab3:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("✅ Aucune donnée manquante détectée.")
    else:
        for col, m in miss.items():
            pct = m / len(df) * 100
            st.progress(min(pct, 100)/100, text=f"{col}: {m} ({pct:.1f}%)")
with tab4:
    num = df.select_dtypes(include='number')
    if num.shape[1]:
        st.dataframe(num.describe(), use_container_width=True)
    else:
        st.info("Aucune colonne numérique détectée.")

# --- Navigation : 2 boutons (Préprocessing / GridSearch) ---
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>🎯 Étape suivante</h3>", unsafe_allow_html=True)

btn_col1, btn_col2, _ = st.columns([1,1,1])
with btn_col1:
    if st.button("⚙️ Aller au Préprocessing", use_container_width=True):
        try:
            st.switch_page("pages/2_preprocessing.py")
        except Exception:
            st.warning("Mets Streamlit à jour (≥1.25) pour activer st.switch_page. Sinon utilise la sidebar.", icon="⚠️")

with btn_col2:
    if st.button("🚀 Passer direct à GridSearch", use_container_width=True):
        # Vérification : dataset réellement prêt pour GridSearch ?
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.error(
                "❌ Le dataset contient encore des **colonnes catégorielles** "
                f"(ex: {cat_cols[:6]}{' ...' if len(cat_cols)>6 else ''}).\n\n"
                "La page GridSearch attend un dataset **déjà encodé** (numérique). "
                "Passe d'abord par **Préprocessing**.", icon="🚫"
            )
        else:
            try:
                st.switch_page("pages/3_Modélisation_et_GridSearch.py")
            except Exception:
                st.warning("Mets Streamlit à jour (≥1.25) pour activer st.switch_page. Sinon utilise la sidebar.", icon="⚠️")
