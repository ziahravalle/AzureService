import io
import os
import tempfile
import uuid
import requests
import streamlit as st
from PIL import Image, ImageDraw

from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential

st.set_page_config(page_title="Azure AI Services Studio", page_icon="☁️", layout="wide")
st.title("☁️ Laboratorio Interactivo de Servicios Azure AI")
st.markdown("Ingresa tus propios datos en cada pestaña para procesarlos en tiempo real con los servicios cognitivos de Azure.")


def get_secret(name: str) -> str:
    """Obtiene credenciales desde st.secrets (nube) o variables de entorno (local)."""
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, "")


tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Language (Sentimientos)",
    "🌐 Translator (Traductor)",
    "🎙️ Speech (Voz ↔ Texto)",
    "👁️ Vision (Detección de Objetos)",
])

# ==============================================================================
# 1. AZURE AI LANGUAGE: ANÁLISIS DE SENTIMIENTO Y MINERIA DE OPINIONES
# ==============================================================================
with tab1:
    st.subheader("Análisis de Sentimiento y Minería de Opiniones")
    col1, col2 = st.columns([3, 1])
    with col1:
        texto_lang = st.text_area(
            "Escribe uno o varios comentarios/reseñas (uno por línea):",
            height=150,
            placeholder="Ej: La comida estuvo deliciosa y la atención fue rápida, aunque el precio es un poco elevado.",
        )
    with col2:
        idioma_lang = st.selectbox(
            "Idioma del texto:",
            options=[("Español", "es"), ("Inglés", "en"), ("Portugués", "pt"), ("Francés", "fr")],
            format_func=lambda x: x[0],
        )[1]
        opinion_mining = st.checkbox("Activar minería de opiniones (aspectos)", value=True)

    if st.button("Analizar Sentimiento", type="primary", key="btn_lang"):
        docs = [line.strip() for line in texto_lang.splitlines() if line.strip()]
        if not docs:
            st.warning("Por favor, escribe al menos una línea de texto para analizar.")
        else:
            try:
                client = TextAnalyticsClient(
                    endpoint=get_secret("LANGUAGE_ENDPOINT"),
                    credential=AzureKeyCredential(get_secret("LANGUAGE_KEY")),
                )
                with st.spinner("Analizando texto con Azure Language..."):
                    resultados = client.analyze_sentiment(
                        docs, language=idioma_lang, show_opinion_mining=opinion_mining
                    )

                for texto, r in zip(docs, resultados):
                    if r.is_error:
                        st.error(f"Error en '{texto}': {r.error.message}")
                        continue

                    s = r.confidence_scores
                    icono = {"positive": "🟢", "neutral": "🟡", "negative": "🔴", "mixed": "🟠"}.get(r.sentiment, "⚪")
                    with st.container(border=True):
                        st.markdown(f"**Texto:** {texto}")
                        st.markdown(
                            f"{icono} **Sentimiento general:** `{r.sentiment.upper()}` "
                            f"(Positivo: `{s.positive:.2f}` | Neutral: `{s.neutral:.2f}` | Negativo: `{s.negative:.2f}`)"
                        )
                        if opinion_mining:
                            aspectos_encontrados = False
                            for oracion in r.sentences:
                                for op in oracion.mined_opinions:
                                    aspectos_encontrados = True
                                    evals = ", ".join(f"'{a.text}' ({a.sentiment})" for a in op.assessments)
                                    st.markdown(f"- **Aspecto `{op.target.text}`** ({op.target.sentiment}): {evals}")
                            if not aspectos_encontrados:
                                st.caption("No se detectaron aspectos específicos evaluados en este texto.")
            except Exception as e:
                st.error(f"Ocurrió un error al conectar con Azure Language: {e}")

# ==============================================================================
# 2. AZURE AI TRANSLATOR: TRADUCCIÓN DE TEXTO
# ==============================================================================
with tab2:
    st.subheader("Traductor Neuronal")
    idiomas_trad = {
        "Español": "es",
        "Inglés": "en",
        "Francés": "fr",
        "Alemán": "de",
        "Portugués": "pt",
        "Italiano": "it",
        "Japonés": "ja",
    }

    c1, c2 = st.columns(2)
    with c1:
        origen_nombre = st.selectbox("Idioma de origen:", list(idiomas_trad.keys()), index=0)
    with c2:
        destinos_nombres = st.multiselect(
            "Idioma(s) de destino:",
            [k for k in idiomas_trad.keys() if k != origen_nombre],
            default=["Inglés"],
        )

    texto_trad = st.text_area(
        "Escribe el texto que deseas traducir:",
        height=130,
        placeholder="Escribe aquí tu párrafo o frase...",
    )

    if st.button("Traducir Texto", type="primary", key="btn_trad"):
        if not texto_trad.strip():
            st.warning("Ingresa un texto para traducir.")
        elif not destinos_nombres:
            st.warning("Selecciona al menos un idioma de destino.")
        else:
            try:
                url = "https://api.cognitive.microsofttranslator.com/translate"
                params = {
                    "api-version": "3.0",
                    "from": idiomas_trad[origen_nombre],
                    "to": [idiomas_trad[d] for d in destinos_nombres],
                }
                headers = {
                    "Ocp-Apim-Subscription-Key": get_secret("TRANSLATOR_KEY"),
                    "Ocp-Apim-Subscription-Region": get_secret("TRANSLATOR_REGION"),
                    "Content-Type": "application/json",
                    "X-ClientTraceId": str(uuid.uuid4()),
                }
                with st.spinner("Traduciendo..."):
                    resp = requests.post(url, params=params, headers=headers, json=[{"text": texto_trad}])
                    resp.raise_for_status()
                    traducciones = resp.json()[0]["translations"]

                for t in traducciones:
                    nombre_dest = next((k for k, v in idiomas_trad.items() if v == t["to"]), t["to"])
                    st.markdown(f"**{nombre_dest} (`{t['to']}`):**")
                    st.info(t["text"])
            except Exception as e:
                st.error(f"Error en Azure Translator: {e}")

# ==============================================================================
# 3. AZURE AI SPEECH: TEXTO A VOZ (TTS) Y VOZ A TEXTO (STT)
# ==============================================================================
with tab3:
    st.subheader("Síntesis y Reconocimiento de Voz")
    col_tts, col_stt = st.columns(2)

    # --- 3.1 Texto a Voz ---
    with col_tts:
        st.markdown("### 🔊 Texto a Voz (TTS)")
        voz = st.selectbox(
            "Selecciona una voz neuronal:",
            options=[
                ("Camila (Español Perú - Mujer)", "es-PE-CamilaNeural"),
                ("Alex (Español Perú - Hombre)", "es-PE-AlexNeural"),
                ("Dalia (Español México - Mujer)", "es-MX-DaliaNeural"),
                ("Elvira (Español España - Mujer)", "es-ES-ElviraNeural"),
                ("Jenny (Inglés EE.UU. - Mujer)", "en-US-JennyNeural"),
            ],
            format_func=lambda x: x[0],
        )[1]
        texto_tts = st.text_area(
            "Escribe lo que quieres que diga la voz:",
            height=110,
            placeholder="Hola, bienvenidos a nuestra aplicación web...",
        )

        if st.button("Generar Audio", type="primary", key="btn_tts"):
            if not texto_tts.strip():
                st.warning("Escribe un texto primero.")
            else:
                try:
                    speech_config = speechsdk.SpeechConfig(
                        subscription=get_secret("SPEECH_KEY"),
                        region=get_secret("SPEECH_REGION"),
                    )
                    speech_config.speech_synthesis_voice_name = voz

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        ruta_wav = tmp_wav.name

                    audio_out = speechsdk.audio.AudioOutputConfig(filename=ruta_wav)
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_out)

                    with st.spinner("Sintetizando voz..."):
                        res_tts = synthesizer.speak_text_async(texto_tts).get()
                    del synthesizer

                    if res_tts.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        with open(ruta_wav, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            "⬇️ Descargar audio WAV",
                            data=audio_bytes,
                            file_name="voz_azure.wav",
                            mime="audio/wav",
                        )
                    else:
                        st.error(f"Error TTS: {res_tts.cancellation_details.error_details}")
                    os.remove(ruta_wav)
                except Exception as e:
                    st.error(f"Error en síntesis de voz: {e}")

    # --- 3.2 Voz a Texto ---
    with col_stt:
        st.markdown("### 🎤 Voz a Texto (STT)")
        idioma_stt = st.selectbox(
            "Idioma del audio a transcribir:",
            options=[
                ("Español (Perú)", "es-PE"),
                ("Español (México)", "es-MX"),
                ("Español (España)", "es-ES"),
                ("Inglés (EE.UU.)", "en-US"),
            ],
            format_func=lambda x: x[0],
        )[1]

        fuente_audio = st.radio(
            "¿Cómo deseas ingresar el audio?",
            ["Grabar con micrófono", "Subir archivo WAV"],
            horizontal=True,
        )

        audio_input_bytes = None
        if fuente_audio == "Grabar con micrófono":
            grabacion = st.audio_input("Graba tu voz (formato WAV):")
            if grabacion is not None:
                audio_input_bytes = grabacion.getvalue()
        else:
            archivo_wav = st.file_uploader("Sube un archivo de audio (.wav)", type=["wav"])
            if archivo_wav is not None:
                audio_input_bytes = archivo_wav.getvalue()

        if st.button("Transcribir Audio", type="primary", key="btn_stt"):
            if audio_input_bytes is None:
                st.warning("Graba o sube un audio en formato WAV primero.")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                        tmp_in.write(audio_input_bytes)
                        ruta_in = tmp_in.name

                    speech_config = speechsdk.SpeechConfig(
                        subscription=get_secret("SPEECH_KEY"),
                        region=get_secret("SPEECH_REGION"),
                    )
                    speech_config.speech_recognition_language = idioma_stt
                    audio_in = speechsdk.audio.AudioConfig(filename=ruta_in)
                    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_in)

                    with st.spinner("Transcribiendo audio..."):
                        res_stt = recognizer.recognize_once_async().get()
                    del recognizer
                    os.remove(ruta_in)

                    if res_stt.reason == speechsdk.ResultReason.RecognizedSpeech:
                        st.success("**Transcripción completada:**")
                        st.write(res_stt.text)
                    elif res_stt.reason == speechsdk.ResultReason.NoMatch:
                        st.warning("No se pudo reconocer voz clara en el audio.")
                    else:
                        st.error(f"Error STT: {res_stt.cancellation_details.error_details}")
                except Exception as e:
                    st.error(f"Error en reconocimiento de voz: {e}")

# ==============================================================================
# 4. AZURE AI VISION: DETECCIÓN DE OBJETOS EN IMÁGENES
# ==============================================================================
with tab4:
    st.subheader("Detección de Objetos en Imágenes")
    archivo_img = st.file_uploader("Sube una fotografía (JPG, JPEG o PNG):", type=["jpg", "jpeg", "png"])

    if archivo_img is not None:
        img_original = Image.open(archivo_img).convert("RGB")
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.image(img_original, caption="Imagen subida por el usuario", use_container_width=True)

        if st.button("Detectar Objetos", type="primary", key="btn_vision"):
            try:
                buf = io.BytesIO()
                img_original.save(buf, format="JPEG")

                vision_client = ImageAnalysisClient(
                    endpoint=get_secret("VISION_ENDPOINT"),
                    credential=AzureKeyCredential(get_secret("VISION_KEY")),
                )

                with st.spinner("Analizando imagen con Azure AI Vision..."):
                    result = vision_client.analyze(
                        image_data=buf.getvalue(),
                        visual_features=[VisualFeatures.OBJECTS],
                    )

                img_anotada = img_original.copy()
                draw = ImageDraw.Draw(img_anotada)

                objetos = result.objects.list if result.objects else []
                for obj in objetos:
                    tag, b = obj.tags[0], obj.bounding_box
                    draw.rectangle([b.x, b.y, b.x + b.width, b.y + b.height], outline="red", width=4)
                    etiqueta = f"{tag.name} ({tag.confidence:.0%})"
                    draw.text((b.x + 6, b.y + 6), etiqueta, fill="red")

                with col_der:
                    st.image(img_anotada, caption="Resultado con objetos detectados", use_container_width=True)

                if objetos:
                    st.markdown("### 📋 Resumen de objetos detectados")
                    datos_tabla = [
                        {
                            "Objeto": obj.tags[0].name,
                            "Confianza": f"{obj.tags[0].confidence:.2%}",
                            "Posición (x, y)": f"({obj.bounding_box.x}, {obj.bounding_box.y})",
                            "Tamaño (w × h)": f"{obj.bounding_box.width} × {obj.bounding_box.height} px",
                        }
                        for obj in objetos
                    ]
                    st.dataframe(datos_tabla, use_container_width=True)
                else:
                    st.info("Azure Vision no detectó objetos reconocibles en esta imagen.")
            except Exception as e:
                st.error(f"Error en Azure Vision: {e}")