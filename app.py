import io
import json
import re
import time
import zipfile
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import types


# =========================
# Config
# =========================
APP_TITLE = "SPRING OS — Direction → Visual Pack™"
TAGLINE = "Dirección primero. Prompts después. Piezas al final."

# Modelos: texto estable por default
TEXT_MODEL_DEFAULT = "gemini-2.5-flash"

# Imagen: por default, NO uses el experimental si tu entrega necesita estabilidad.
# Dejamos el experimental como opción manual si querés.
IMAGE_MODEL_DEFAULT = "imagen-3.0-generate-002"  # más estable en general para imagen
# Alternativa (experimental / cuotas volátiles): gemini-2.5-flash-image

TIMEOUT_HINT = "Si ves 429 en imágenes, suele ser cuota/billing del proyecto (no bug)."

MAX_PROMPTS_HARD = 10
MAX_RETRIES_429 = 3


# =========================
# Utils
# =========================
def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúñü\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:60] or "spring"


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\{.*\})", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def _zip_files(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


def _aspect_from_format(fmt_label: str) -> str:
    return {
        "Reel/Story (9:16)": "9:16",
        "Post (1:1)": "1:1",
        "Horizontal (16:9)": "16:9",
    }[fmt_label]


def _must_have(bp: Dict[str, Any], key: str) -> Optional[str]:
    if key not in bp:
        return f"Falta: {key}"
    return None


def validate_prompt_pack(pack: Dict[str, Any]) -> List[str]:
    issues = []
    if not isinstance(pack, dict):
        return ["La salida no es un JSON objeto."]

    for k in ["brand_snapshot", "prompts", "notes"]:
        e = _must_have(pack, k)
        if e:
            issues.append(e)

    prompts = pack.get("prompts")
    if not isinstance(prompts, list) or len(prompts) == 0:
        issues.append("prompts debe ser lista con al menos 1 elemento.")
        return issues

    for i, p in enumerate(prompts, start=1):
        if not isinstance(p, dict):
            issues.append(f"prompts[{i}] no es objeto.")
            continue
        for rk in ["id", "piece_type", "format", "aspect_ratio", "prompt"]:
            if rk not in p or not str(p.get(rk, "")).strip():
                issues.append(f"prompts[{i}] falta {rk} o está vacío.")
        ar = p.get("aspect_ratio", "")
        if ar not in ("9:16", "1:1", "16:9"):
            issues.append(f"prompts[{i}] aspect_ratio inválido: {ar}")

    return issues


def _hash_pack(pack: Dict[str, Any]) -> str:
    raw = json.dumps(pack, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# =========================
# Google (Direct calls)
# =========================
def google_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def google_text_json(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float = 0.4,
    max_output_tokens: int = 4096,
) -> Tuple[Dict[str, Any], str]:
    resp = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    text = getattr(resp, "text", "") or ""
    parsed = _safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("Gemini devolvió texto, pero no pude parsear JSON.")
    return parsed, text


def google_text_fix_json(
    client: genai.Client,
    model: str,
    bad_output: str,
) -> Dict[str, Any]:
    fix_prompt = f"""
Tu salida anterior NO fue JSON válido o no cumplió el esquema.
Devolveme SOLAMENTE un JSON válido (sin markdown, sin texto extra).

Salida anterior:
{bad_output}
""".strip()
    parsed, _ = google_text_json(client, model, fix_prompt, temperature=0.2, max_output_tokens=4096)
    return parsed


def _extract_image_bytes_from_response(resp) -> Optional[bytes]:
    """
    Robust: intenta extraer bytes de imagen con distintas formas del SDK.
    Prioridad:
    - part.inline_data.data si mime_type image/*
    - part.data si existe
    - part.as_image() si existe (y luego serialize)
    """
    try:
        parts = resp.candidates[0].content.parts
    except Exception:
        return None

    for part in parts:
        # 1) inline_data
        try:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                mime = getattr(inline, "mime_type", "") or ""
                if mime.startswith("image/"):
                    return inline.data
        except Exception:
            pass

        # 2) data directo
        try:
            data = getattr(part, "data", None)
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                return bytes(data)
        except Exception:
            pass

        # 3) as_image fallback
        try:
            if hasattr(part, "as_image"):
                img = part.as_image()
                buf = io.BytesIO()
                # Pillow-like object usually supports save()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            pass

    return None


def google_generate_image_bytes(
    client: genai.Client,
    model: str,
    prompt: str,
    aspect_ratio: str,
) -> bytes:
    """
    Retry loop correcto:
    - Reintenta SOLO en 429/RESOURCE_EXHAUSTED.
    - En el último intento, si sigue 429, levanta error limpio.
    - Cualquier otro error: levanta inmediato (sin loop raro).
    """
    last_err = None

    for attempt in range(MAX_RETRIES_429 + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )

            img_bytes = _extract_image_bytes_from_response(resp)
            if not img_bytes:
                raise RuntimeError("No llegó imagen (bytes vacíos o formato inesperado).")

            return img_bytes

        except Exception as e:
            last_err = e
            msg = str(e)

            is_429 = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg)

            if is_429 and attempt < MAX_RETRIES_429:
                # backoff exponencial corto
                time.sleep(2 ** attempt)
                continue

            # Si es 429 y ya no hay más intentos, o si NO es 429: abortar limpio
            raise RuntimeError(msg)

    # Nunca debería llegar acá, pero por las dudas:
    raise RuntimeError(str(last_err) if last_err else "Error desconocido generando imagen.")


# =========================
# Prompt builder (Pack)
# =========================
def build_pack_request(brief: Dict[str, Any], pieces: List[str], formats: List[str], max_prompts: int) -> str:
    fmt_aspects = {f: _aspect_from_format(f) for f in formats}

    schema = {
        "brand_snapshot": {
            "one_liner": "string",
            "tone_rules": ["string", "string", "string"],
            "palette_hint": {"primary": "#RRGGBB", "secondary": "#RRGGBB", "accent": "#RRGGBB"},
            "style_words": ["string", "string", "string"],
            "avoid": ["string", "string", "string"],
        },
        "prompts": [
            {
                "id": "string",
                "piece_type": "string",
                "format": "string",
                "aspect_ratio": "9:16|1:1|16:9",
                "prompt": "string",
            }
        ],
        "notes": ["string", "string"],
    }

    return f"""
Eres un estratega creativo senior. Produces piezas reales, no “ideas”.
Idioma: español neutro. Estilo: profesional, cercano, claro.
Prohibido: hype, claims mágicos, “viral”, “garantizado”.

OBJETIVO:
Con este brief, genera un pack de prompts para un generador de imágenes.
Devuelve SOLAMENTE un JSON válido (sin markdown, sin texto extra).

BRIEF:
- Proyecto: {brief["project"]}
- Prioridad del mes: {brief["priority"]}
- Frecuencia real: {brief["freq"]}
- Tono: {brief["tone"]}
- Oferta: {brief["offer"]}
- Promo/campaña (si aplica): {brief["promo"] or "—"}
- Estilo visual: {brief["visual_style"]}
- Paleta referencia: {brief["palette_hint"]}
- Incluir si aplica: {brief["must_include"] or "—"}
- Evitar: {brief["avoid"]}

Piezas elegidas: {pieces}
Formatos + aspect ratio: {fmt_aspects}

REGLAS:
- Máximo {max_prompts} prompts totales.
- Cada prompt debe describir: composición, estilo, iluminación, espacio para texto (si aplica).
- Texto dentro de la imagen: máximo 6 palabras, legible, sin saturar.
- Nada de estética “template barata”.
- Mantener coherencia con tono y paleta.
- Cada item en "prompts" debe traer aspect_ratio correcto.

ESQUEMA (respétalo):
{json.dumps(schema, ensure_ascii=False)}
""".strip()


# =========================
# Streamlit State (anti-reruns)
# =========================
def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "brief" not in st.session_state:
        st.session_state.brief = {}
    if "pack" not in st.session_state:
        st.session_state.pack = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "images" not in st.session_state:
        st.session_state.images = []
    if "images_pack_hash" not in st.session_state:
        st.session_state.images_pack_hash = ""
    if "is_generating_images" not in st.session_state:
        st.session_state.is_generating_images = False


init_state()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
st.title(APP_TITLE)
st.caption(TAGLINE)

with st.sidebar:
    st.markdown("### Keys")
    google_key = st.text_input("Google API Key", type="password", placeholder="Pegá tu key de Google")

    st.markdown("### Modelos")
    text_model = st.text_input("Modelo Texto (cerebro)", value=TEXT_MODEL_DEFAULT)
    image_model = st.text_input("Modelo Imagen", value=IMAGE_MODEL_DEFAULT)

    st.markdown("---")
    st.caption(TIMEOUT_HINT)


# STEP 1
if st.session_state.step == 1:
    st.markdown("## Punto de partida")
    st.info("Definamos lo mínimo. Con esto alcanza para un plan claro. Si querés, después afinamos.")

    project = st.text_input("Proyecto", value="SPRING")

    priority = st.selectbox(
        "Prioridad del mes (una sola)",
        ["Venta", "Promo puntual", "Branding (marca)", "Comunidad (interacción y vínculo)"],
        index=2,
    )

    freq = st.selectbox("Frecuencia real", ["2/semana", "3/semana", "diario"], index=1)
    st.caption("Cumplir > fantasear.")

    tone = st.selectbox(
        "Tono de mensajes",
        [
            "Estratégico y claro",
            "Cercano y didáctico",
            "Sofisticado y minimalista",
            "Directo y ejecutivo",
            "Inspirador pero realista",
        ],
        index=0,
    )

    offer = st.text_input("Oferta (1 línea)", value="Servicio / consultoría / producto digital")
    promo = st.text_input("Promo/campaña (si aplica)", value="")

    st.markdown("#### Dirección visual (rápida)")
    visual_style = st.selectbox(
        "Estilo",
        [
            "Tech claro (limpio, ordenado, UI-ish)",
            "Minimalista premium (aireado, editorial)",
            "Vibrante pro (energía con control)",
            "Cálido humano (simple, sin infantilizar)",
            "Moderno disruptivo (composición audaz)",
        ],
        index=0,
    )

    palette_hint = st.text_input("Paleta (referencia)", value="rosa suave + verde oscuro + beige (sofisticado)")
    must_include = st.text_input("Incluir (si aplica)", value="")
    avoid = st.text_input("Evitar", value="exceso de texto, template barato, colores sin control")

    if st.button("Siguiente →", type="primary", use_container_width=True):
        st.session_state.brief = {
            "project": project.strip(),
            "priority": priority,
            "freq": freq,
            "tone": tone,
            "offer": offer.strip(),
            "promo": promo.strip(),
            "visual_style": visual_style,
            "palette_hint": palette_hint.strip(),
            "must_include": must_include.strip(),
            "avoid": avoid.strip(),
        }
        st.session_state.pack = None
        st.session_state.images = []
        st.session_state.images_pack_hash = ""
        st.session_state.raw_text = ""
        st.session_state.step = 2
        st.rerun()


# STEP 2
if st.session_state.step == 2:
    brief = st.session_state.brief
    st.markdown("## 2) Armemos los prompts (Gemini texto)")

    st.write(
        f"**Proyecto:** {brief['project']} · **Prioridad:** {brief['priority']} · **Frecuencia:** {brief['freq']} · **Tono:** {brief['tone']}"
    )

    pieces = st.multiselect(
        "Tipos de piezas",
        [
            "Post promo (oferta / precio)",
            "Post educativo (tip rápido)",
            "Post prueba social (testimonio / resultado)",
            "Story 1 (gancho)",
            "Story 2 (explicación corta)",
            "Story 3 (CTA)",
            "Portada highlight",
        ],
        default=["Post promo (oferta / precio)", "Post educativo (tip rápido)"],
    )

    formats = st.multiselect(
        "Formatos",
        ["Reel/Story (9:16)", "Post (1:1)", "Horizontal (16:9)"],
        default=["Reel/Story (9:16)", "Post (1:1)"],
    )

    max_prompts = st.slider("Cantidad máxima de prompts", 2, MAX_PROMPTS_HARD, 6)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Volver", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with c2:
        gen_prompts = st.button("Generar prompts", type="primary", use_container_width=True)

    if gen_prompts:
        if not google_key:
            st.error("Te falta la Google API Key.")
            st.stop()
        if not pieces or not formats:
            st.error("Elegí al menos 1 tipo de pieza y 1 formato.")
            st.stop()

        client = google_client(google_key)
        req = build_pack_request(brief, pieces, formats, max_prompts)

        with st.spinner("Armando prompts…"):
            try:
                pack, raw = google_text_json(
                    client=client,
                    model=text_model.strip(),
                    prompt=req,
                    temperature=0.35,
                    max_output_tokens=4096,
                )
                st.session_state.raw_text = raw

                issues = validate_prompt_pack(pack)
                if issues:
                    pack = google_text_fix_json(client, text_model.strip(), raw)

            except Exception as e:
                st.error(f"No pude generar prompts: {e}")
                st.stop()

        issues = validate_prompt_pack(pack)
        if issues:
            st.error("Salida inválida:")
            for it in issues:
                st.write(f"- {it}")
            st.caption("Tip: bajá la cantidad de prompts o simplificá el brief.")
            st.stop()

        pack["prompts"] = pack["prompts"][:max_prompts]
        st.session_state.pack = pack

        # Protección: al cambiar pack, invalidamos imágenes anteriores
        st.session_state.images = []
        st.session_state.images_pack_hash = _hash_pack(pack)

        st.success("Listo. Prompts generados.")
        st.session_state.step = 3
        st.rerun()


# STEP 3
if st.session_state.step == 3:
    pack = st.session_state.pack
    if not pack:
        st.error("No hay prompts generados. Volvé al paso 2.")
        st.stop()

    st.markdown("## 3) Generación del Visual Pack™")
    st.caption("Si imagen tiene cuota 0, igual podés descargar el JSON de prompts como evidencia de la llamada IA.")

    # Snapshot
    brand = pack.get("brand_snapshot", {})
    if brand:
        st.markdown("### Snapshot")
        if brand.get("one_liner"):
            st.write(brand["one_liner"])
        pal = brand.get("palette_hint", {})
        st.caption(f"Paleta sugerida: {pal.get('primary','—')} · {pal.get('secondary','—')} · {pal.get('accent','—')}")

    st.markdown("### Prompts listos")
    for p in pack.get("prompts", []):
        st.write(f"- **{p.get('piece_type','')}** · {p.get('format','')} · {p.get('aspect_ratio','')}")
        st.code(p.get("prompt", ""), language="text")

    st.download_button(
        "Descargar prompts (JSON)",
        data=json.dumps(pack, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"prompts_{_now_id()}.json",
        mime="application/json",
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Volver", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with c2:
        gen_images = st.button(
            "Generar imágenes",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_generating_images,
        )

    # Generación protegida: solo corre cuando apretás el botón (y con flag anti-doble)
    if gen_images:
        if not google_key:
            st.error("Te falta la Google API Key.")
            st.stop()

        st.session_state.is_generating_images = True

        # Si ya hay imágenes generadas para este pack, NO regenerar
        current_hash = _hash_pack(pack)
        if st.session_state.images and st.session_state.images_pack_hash == current_hash:
            st.info("Ya tenés imágenes generadas para este pack. No regenero.")
            st.session_state.is_generating_images = False
            st.stop()

        client = google_client(google_key)
        results: List[Tuple[str, bytes]] = []

        try:
            with st.spinner("Generando activos con IA…"):
                for idx, p in enumerate(pack.get("prompts", []), start=1):
                    piece = p.get("piece_type", "pieza")
                    fmt = p.get("format", "formato")
                    aspect = p.get("aspect_ratio", "1:1")
                    prompt_text = p.get("prompt", "")

                    if not prompt_text.strip():
                        continue

                    img_bytes = google_generate_image_bytes(
                        client=client,
                        model=image_model.strip(),
                        prompt=prompt_text,
                        aspect_ratio=aspect,
                    )

                    filename = f"{_slug(piece)}_{_slug(fmt)}_{idx:02d}_{aspect.replace(':','x')}.png"
                    results.append((filename, img_bytes))

            st.session_state.images = results
            st.session_state.images_pack_hash = current_hash
            st.success("Listo. Imágenes generadas.")

        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                st.error("Límite de cuota (429). Tu proyecto no tiene cuota habilitada o se quedó sin quota para IMAGEN.")
                st.caption("Solución típica: habilitar billing / revisar cuotas del modelo de imagen en tu proyecto.")
            else:
                st.error(f"Error generando imágenes: {e}")

        finally:
            st.session_state.is_generating_images = False

    images = st.session_state.images
    if images:
        st.markdown("### Preview")
        cols = st.columns(3)
        for i, (fn, b) in enumerate(images[:9]):
            with cols[i % 3]:
                st.image(b, caption=fn, use_container_width=True)

        zip_bytes = _zip_files(images)
        st.download_button(
            "Descargar pack (.zip)",
            data=zip_bytes,
            file_name=f"visual_pack_{_now_id()}.zip",
            mime="application/zip",
            use_container_width=True,
        )
