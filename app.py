import io
import time
import zipfile
import streamlit as st
from google import genai
from google.genai import types

# CONFIGURACIÓN PARA CUENTA PAGA
APP_TITLE = "SPRING OS — Visual Pack™ (Paid Tier)"
# Gemini 2.0 Flash es el motor recomendado para alta velocidad y multimedia
MODEL_ENGINE = "gemini-2.0-flash" 

def _zip_images(images):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, b in images:
            z.writestr(fn, b)
    return buf.getvalue()

def _generate_image_paid(api_key, prompt, aspect_ratio):
    client = genai.Client(api_key=api_key)
    # Reintento agresivo pero seguro para evitar el bloqueo por frecuencia
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL_ENGINE,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
            if resp.candidates and resp.candidates[0].content.parts:
                img = resp.candidates[0].content.parts[0].as_image()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            if "429" in str(e):
                # En cuenta paga, una pausa de 5-10s suele ser suficiente para resetear el RPM
                time.sleep(10)
            else:
                raise e
    raise RuntimeError("Error de cuota persistente. Revisa tu consola de Google Cloud.")

# --- INTERFAZ ---
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    api_key = st.text_input("Google API Key (Pago)", type="password")

if "step" not in st.session_state: st.session_state.step = 1

if st.session_state.step == 1:
    project = st.text_input("Proyecto", "SPRING")
    if st.button("Siguiente"):
        st.session_state.data = {"project": project}
        st.session_state.step = 2
        st.rerun()

if st.session_state.step == 2:
    # Ahora que pagas, podemos subir el límite a 12 de nuevo con seguridad
    num = st.slider("Cantidad de imágenes (Pack Completo)", 1, 12, 6)
    fmt = st.selectbox("Formato", ["Post (1:1)", "Story (9:16)"])
    
    if st.button("🚀 Generar Visual Pack™"):
        if not api_key: st.error("Falta API Key"); st.stop()
        
        outputs = []
        progress = st.progress(0)
        with st.spinner("Generando piezas a alta velocidad..."):
            for i in range(num):
                try:
                    aspect = "1:1" if "1:1" in fmt else "9:16"
                    prompt = f"Professional high-end brand visual for {st.session_state.data['project']}. Cinematic lighting."
                    img_b = _generate_image_paid(api_key, prompt, aspect)
                    outputs.append((f"v{i+1}.png", img_b))
                    progress.progress((i + 1) / num)
                    # Pausa mínima de seguridad para no saturar el RPM de pago
                    time.sleep(2) 
                except Exception as e:
                    st.error(f"Fallo en imagen {i+1}: {e}"); break
        
        st.session_state.outputs = outputs

    if "outputs" in st.session_state:
        st.download_button("⬇️ Descargar Pack (.zip)", _zip_images(st.session_state.outputs), "spring_pack.zip")
