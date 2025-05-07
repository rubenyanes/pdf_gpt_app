import streamlit as st
import os
import pandas as pd
import base64
from PIL import Image
from functions import process_pdfs_in_folder
from io import BytesIO
import tempfile
import json
import uuid
import fitz

# ---------------------- Limpieza forzada del session_state para evitar errores de deserialización ----------------------
for k in list(st.session_state.keys()):
    if "upload" in k or "file" in k:
        try:
            del st.session_state[k]
        except:
            pass
unique_id = str(uuid.uuid4())[:8]

# ---------------------- Configurazione della pagina ----------------------
st.set_page_config(page_title="Estrazione Tecnica", layout="wide")

# ---------------------- Reset dello stato se l'elaborazione è stata fermata ----------------------
if st.session_state.get('processing_stopped', False):
    keys_to_reset = ['stop_process', 'start_process', 'processing_stopped']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# ---------------------- Stile custom e logo ----------------------
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6, label, .stTextInput > label, .stTextArea > label, .stDownloadButton, .stButton > button {
        color: #003366;
    }
    .centered-text {
        display: flex;
        justify-content: center;
        text-align: center;
        width: 100%;
    }
    .logo-small {
        width: 120px;
        margin-bottom: 10px;
    }
    .button-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Mostra logo
logo_path = os.path.join(os.path.dirname(__file__), "logo_sinco.png")
logo_image = Image.open(logo_path)
buffered = base64.b64encode(open(logo_path, "rb").read()).decode()

st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 1rem;'>
        <img src='data:image/png;base64,{buffered}' class='logo-small'>
    </div>
""", unsafe_allow_html=True)

# ---------------------- Titolo e descrizione ----------------------
st.markdown("""
    <div class='centered-text'>
        <h1 style='color: #003366;'>Estrazione di materiali impiegati e membratura delle prove</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='centered-text'>Estrai automaticamente i dati tecnici (Spessore e Materiale di Fasciame e Fondo) da specifiche PDF usando OCR + Openai GPT-4o</div>", unsafe_allow_html=True)

# ---------------------- Separatore visivo ----------------------
st.markdown("<hr style='margin-top:1.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# ---------------------- Inputs utente ----------------------
st.markdown("📂 **Carica uno o più file PDF**", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Carica uno o più file PDF", type=["pdf","PDF"], accept_multiple_files=True, label_visibility="collapsed", key=f"upload_pdfs_{unique_id}")

# ---------------------- Validación de PDF ----------------------
valid_files = []
if uploaded_files:
    for file in uploaded_files:
        try:
            file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            fitz.open(tmp_path)  # Validación básica
            valid_files.append(file)
        except Exception as e:
            st.warning(f"⚠️ Il file `{file.name}` non è un PDF valido: {e}")
if uploaded_files and not valid_files:
    st.stop()

uploaded_files = valid_files

# ---------------------- Caricamento credenziali Google Vision ----------------------
st.markdown("🔐 **Carica il file di credenziali JSON di Google Cloud Vision**")
uploaded_cred = st.file_uploader("Carica il file di credenziali JSON", type=["json","JSON"], label_visibility="collapsed", key=f"upload_cred_{unique_id}")

if uploaded_cred is not None:
    try:
        creds_data = json.load(uploaded_cred)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmpfile:
            json.dump(creds_data, tmpfile)
            tmpfile.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmpfile.name
        st.success("✅ Credenziali caricate correttamente!")
    except Exception as e:
        st.error(f"❌ Errore nel caricamento delle credenziali: {e}")
else:
    st.warning("⚠️ Devi caricare il file delle credenziali per poter usare Google Vision OCR.")

# ---------------------- OpenAI API Key ----------------------
st.markdown("🔑 **OpenAI Api key** (https://platform.openai.com/api-keys)")
openai_key = st.text_input("Inserisci la tua API Key", type="password")

# ---------------------- Keywords e valid values ----------------------
default_keywords = [
    'fasciame', 'fondo', 'fondi', 'materiali impiegati', 'spessore', 'KW', 'prove', 'Fe',
    'calotta', 'tronchetti', 'membratura','costruttore','temp','temperatura', 'qualita',
    'talloni di saldatura','nominale'
]
keywords = st.text_area("🔍 Parole chiave per la rilevazione", "\n".join(default_keywords)).splitlines()

default_valid_values = [
    "Fe 510,2KW", "Fe 510,2 KG", "Fe 410,2 KW", "Fe 410,2 KG",
    "P 355 N", "Fe 52/2", "Fe 52/c", "Fe 42/c", "Fe 42/d",
    "Fe 460,2KW", "Fe 360,2KG", "Fe 460,2KG",
    "Fe 44/c", "Fe 42/I", "Fe 52/D", "Fe 44/D"
]
valid_values = st.text_area("✅ Valori validi accettati per i materiali", "\n".join(default_valid_values)).splitlines()

# ---------------------- Pulsanti per avviare e fermare ----------------------
col1, col2, _ = st.columns([1, 1, 6])
with col1:
    if st.button("🚀 Avvia Elaborazione"):
        st.session_state['start_process'] = True
        st.session_state['stop_process'] = False

with col2:
    if st.button("🛑 Ferma Elaborazione"):
        st.session_state['stop_process'] = True
        st.session_state['start_process'] = False
        st.session_state['processing_stopped'] = True

# ---------------------- Logica di elaborazione ----------------------
if st.session_state.get('start_process', False):
    if not uploaded_files or not openai_key or not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error("❌ Devi caricare almeno un PDF, inserire la tua API key di OpenAI e le credenziali di Google Vision.")
        st.session_state['start_process'] = False
    else:
        with st.spinner("Elaborazione dei PDF in corso..."):
            progress_placeholder = st.empty()

            def stream_progress(idx, total):
                if st.session_state.get('stop_process', False):
                    raise RuntimeError("⚠️ Elaborazione interrotta dall'utente.")
                progress_placeholder.markdown(f"### ⏳ Elaborazione: {idx}/{total} PDF")

            try:
                df = process_pdfs_in_folder(uploaded_files, keywords, valid_values, openai_key, stream_progress)

                df = df.rename(columns={
                    "Nome": "Nome PDF",
                    "Fasciame Spessore": "Spessore Fasciame",
                    "Qualita Fasciame": "Materiale Fasciame",
                    "Fondo Spessore": "Spessore Fondo",
                    "Qualita Fondo": "Materiale Fondo"
                })
                df["Libretto"] = "Sì"
                df = df[[
                    "Nome PDF",
                    "Spessore Fasciame",
                    "Materiale Fasciame",
                    "Spessore Fondo",
                    "Materiale Fondo",
                    "Libretto"
                ]]

                st.success("✅ Elaborazione completata!")
                st.dataframe(df, use_container_width=True, hide_index=True)

                output = BytesIO()
                df.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)

                st.download_button(
                    label="📥 Scarica il file Excel",
                    data=output,
                    file_name="risultato.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except RuntimeError as e:
                st.warning(str(e))
                progress_placeholder.markdown("### ⛔ Elaborazione interrotta")

        st.session_state['start_process'] = False
