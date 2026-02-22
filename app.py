import io
import time
import zipfile
import streamlit as st
from google import genai
from google.genai import types

# CONFIGURACIÓN PARA CUENTA PAGA (TIER 1)
APP_TITLE = "SPRING OS — Visual Pack™ (Final)"
# Cambiamos a 'gemini-1.5-flash' para texto y 'imagen-3.0-generate-001' para imagen
# Estos son los modelos que SÍ aceptan peticiones hoy sin dar error 404.
TEXT_MODEL = "gemini-1.5-flash"
IMAGE_MODEL = "imagen-3.0-generate-001" 

def _zip_images(images):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, b in images:
            z.writestr(fn, b)
    return buf.getvalue()

def _generate_resilient_image(api_key, prompt, aspect_ratio):
    client = genai.Client(api_key=api_key)
    # Reintento ante error 429 (Límite de frecuencia)
    for attempt in range(3):
        try:
            # IMPORTANTE: Usamos el método correcto para Imagen 3 en cuentas pagas
            resp = client.models.generate_content(
                model=IMAGE_MODEL,
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
                time.sleep(15) # Pausa necesaria para resetear el RPM
            else:
                raise e
    raise RuntimeError("No se pudo generar la imagen.")

# --- INTERFAZ ---
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    api_key = st.text_input("Google API Key (Nivel 1)", type="password")

if "step" not in st.session_state: st.session_state.step = 1

if st.session_state.step == 1:
    project = st.text_input("Proyecto", "SPRING")
    if st.button("Siguiente"):
        st.session_state.data = {"project": project}
        st.session_state.step = 2
        st.rerun()

if st.session_state.step == 2:
    st.header("Generación de Pack")
    # Limitamos a 4 imágenes para asegurar estabilidad total en una ráfaga
    num = st.slider("Cantidad de imágenes", 1, 4, 2)
    
    if st.button("🚀 Iniciar Generación"):
        if not api_key: st.error("Falta API Key"); st.stop()
        
        outputs = []
        with st.spinner("Generando activos de alta calidad..."):
            for i in range(num):
                try:
                    prompt = f"Professional high-end brand visual for {st.session_state.data['project']}. Minimalist style."
                    img_b = _generate_resilient_image(api_key, prompt, "1:1")
                    outputs.append((f"v{i+1}.png", img_b))
                    # Pausa de cortesía para no saturar tu RPM de 1,000
                    time.sleep(5) 
                except Exception as e:
                    st.error(f"Error en imagen {i+1}: {e}"); break
        
        st.session_state.outputs = outputs

    if "outputs" in st.session_state:
        st.success("¡Pack visual listo!")
        st.download_button("Descargar Pack (.zip)", _zip_images(st.session_state.outputs), "pack.zip")
