
import os
import hashlib
import html
import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="The Vinyl Library",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta visual
CREAM = "#F5F0E6"
PAPER = "#E8DECC"
INK = "#242321"
MUTED = "#766D60"
TERRACOTTA = "#8A3D30"
GOLD = "#B99A61"


# --------------------------------------------------
# ESTILO VINTAGE
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');

:root {
    --cream: #F5F0E6;
    --paper: #E8DECC;
    --ink: #242321;
    --muted: #766D60;
    --red: #8A3D30;
    --gold: #B99A61;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: 'Playfair Display', Georgia, serif;
}

header[data-testid="stHeader"] {
    background: rgba(245, 240, 230, 0.94);
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.8px;
}

p, label, .stMarkdown, .stCaption {
    color: var(--ink);
}

.vinyl-kicker {
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 10px;
    color: var(--red);
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(42px, 6vw, 76px);
    font-weight: 400;
    line-height: 1.04;
    letter-spacing: -3px;
    color: var(--ink);
    margin: 0 0 18px 0;
}

.hero-title em {
    color: var(--red);
    font-weight: 400;
}

.hero-description {
    color: var(--muted);
    font-size: 17px;
    line-height: 1.8;
    max-width: 620px;
}

.vinyl-scene {
    background: var(--paper);
    min-height: 270px;
    height: 100%;
    border: 1px solid #D4C7B1;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.vinyl-disc {
    width: 205px;
    height: 205px;
    border-radius: 50%;
    background:
        repeating-radial-gradient(
            circle at center,
            #171716 0px,
            #252523 2px,
            #10100F 4px,
            #2C2C29 5px,
            #141413 7px
        );
    box-shadow: 9px 14px 25px #8E816B80;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: vinyl-spin 24s linear infinite;
}

.vinyl-label {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: var(--red);
    border: 4px double #C5A66D;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #F5EBDD;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 1px;
    text-align: center;
}

.vinyl-label::after {
    content: '';
    width: 5px;
    height: 5px;
    background: #E7D8BD;
    border-radius: 50%;
    position: absolute;
}

@keyframes vinyl-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.section-label {
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 10px;
    color: var(--red);
    border-bottom: 1px solid #D4C7B1;
    padding-bottom: 12px;
    margin: 22px 0 16px 0;
}

.stButton > button {
    background: var(--ink);
    color: var(--cream);
    border: 1px solid var(--ink);
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 12px 18px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: var(--red);
    color: white;
    border-color: var(--red);
}

.stTextArea textarea,
.stTextInput input {
    background: #FBF8F1;
    color: var(--ink);
    border: 1px solid #D4C7B1;
    border-radius: 2px;
    font-family: 'Playfair Display', Georgia, serif;
}

[data-testid="stFileUploader"] {
    background: #FBF8F1;
    border: 1px dashed #B9A98D;
    border-radius: 3px;
    padding: 15px;
}

[data-testid="stSidebar"] {
    background: #EAE1D1;
    border-right: 1px solid #D4C7B1;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
}

[data-testid="stMetric"] {
    background: #FBF8F1;
    border: 1px solid #D4C7B1;
    border-radius: 3px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: var(--red);
    font-family: 'Playfair Display', Georgia, serif;
}

.stAlert {
    border-radius: 3px;
}

.footer-vinyl {
    border-top: 1px solid #D4C7B1;
    margin-top: 60px;
    padding-top: 20px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    line-height: 2;
    letter-spacing: 1px;
    color: var(--muted);
}

@media (prefers-reduced-motion: reduce) {
    .vinyl-disc {
        animation: none;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PORTADA
# --------------------------------------------------

hero_left, hero_right = st.columns([1.45, 1], gap="large")

with hero_left:
    st.markdown("""
    <div style="padding-top: 20px;">
        <div class="vinyl-kicker">
            THE ANALOG EXPERIENCE · EST. 2026
        </div>
        <div class="hero-title">
            The Vinyl<br>
            <em>Library.</em>
        </div>
        <div class="hero-description">
            Cada documento tiene una historia.
            Cada pregunta, una nueva forma de escucharla.
            Explora tus archivos como quien descubre un
            disco olvidado en una tienda de música.
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("""
    <div class="vinyl-scene">
        <div class="vinyl-disc">
            <div class="vinyl-label">SIDE A<br>THE VINYL<br>LIBRARY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">01 — PREPARA TU SESIÓN DE ESCUCHA</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

with st.sidebar:
    st.markdown("## The listening room")
    st.caption("Tu espacio personal de exploración documental.")

    st.markdown("---")

    st.markdown("### Antes de comenzar")
    st.write(
        "1. Conecta tu clave de OpenAI.\n\n"
        "2. Introduce un PDF.\n\n"
        "3. Haz una pregunta y descubre lo que contiene."
    )

    st.markdown("---")

    st.markdown("### Nuestra colección")
    st.write("01 · Vinyl & analog culture")
    st.write("02 · Music history")
    st.write("03 · Artists & iconic albums")
    st.write("04 · Emotions & musical identity")

    st.markdown("---")

    st.caption(
        "Una experiencia inspirada en los formatos físicos, "
        "las tiendas de discos y la cultura musical."
    )

    api_key = st.text_input(
        "OPENAI API KEY",
        type="password",
        help="Tu clave se utiliza para realizar las consultas a OpenAI."
    )


# --------------------------------------------------
# CARGA DEL DOCUMENTO
# --------------------------------------------------

st.markdown("### Coloca tu próximo disco")
st.write(
    "Sube un documento PDF. Puede ser sobre música, artistas, "
    "historia cultural o cualquier tema que quieras investigar."
)

pdf = st.file_uploader(
    "Selecciona tu documento",
    type=["pdf"],
    help="Formatos admitidos: archivos PDF con texto seleccionable."
)

if pdf is not None:
    st.caption(f"Documento seleccionado: {pdf.name}")

if not api_key:
    st.info(
        "Introduce tu clave de OpenAI en la barra lateral "
        "para activar la biblioteca inteligente."
    )


# --------------------------------------------------
# PROCESAMIENTO Y MEMORIA DE SESIÓN
# --------------------------------------------------

if "vinyl_kb_id" not in st.session_state:
    st.session_state.vinyl_kb_id = None

if "vinyl_knowledge_base" not in st.session_state:
    st.session_state.vinyl_knowledge_base = None

if "vinyl_chunks" not in st.session_state:
    st.session_state.vinyl_chunks = 0

if "vinyl_doc_name" not in st.session_state:
    st.session_state.vinyl_doc_name = None


if pdf is not None and api_key:
    pdf_bytes = pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # El índice se reconstruye solo si cambia el PDF o la clave.
    session_id = hashlib.sha256(
        (file_hash + api_key).encode("utf-8")
    ).hexdigest()

    if st.session_state.vinyl_kb_id != session_id:
        if st.button("Preparar mi disco ↗", type="primary"):
            try:
                with st.spinner(
                    "Leyendo el documento y organizando la colección..."
                ):
                    reader = PdfReader(pdf)
                    pages_text = []

                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages_text.append(page_text)

                    full_text = "\n\n".join(pages_text)

                    if not full_text.strip():
                        st.error(
                            "No se pudo extraer texto de este PDF. "
                            "Si es un documento escaneado, necesitarás "
                            "aplicar OCR antes de cargarlo."
                        )
                        st.stop()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=900,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )

                    chunks = splitter.split_text(full_text)

                    embeddings = OpenAIEmbeddings(
                        api_key=api_key
                    )

                    knowledge_base = FAISS.from_texts(
                        texts=chunks,
                        embedding=embeddings
                    )

                    st.session_state.vinyl_knowledge_base = knowledge_base
                    st.session_state.vinyl_kb_id = session_id
                    st.session_state.vinyl_chunks = len(chunks)
                    st.session_state.vinyl_doc_name = pdf.name

                st.success("¡Tu disco está listo para explorar!")

            except Exception as e:
                st.error(
                    "No se pudo preparar el documento. "
                    "Revisa la clave de API, la conexión y las "
                    "dependencias instaladas."
                )
                st.caption(f"Detalle técnico: {str(e)}")


# --------------------------------------------------
# PANEL DE EXPLORACIÓN
# --------------------------------------------------

if (
    pdf is not None
    and api_key
    and st.session_state.vinyl_kb_id is not None
    and st.session_state.vinyl_kb_id == session_id
    and st.session_state.vinyl_knowledge_base is not None
):
    st.markdown("---")

    st.markdown(
        '<div class="section-label">'
        '02 — EXPLORA LAS NOTAS DE TU DOCUMENTO'
        '</div>',
        unsafe_allow_html=True
    )

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Fragmentos indexados",
            st.session_state.vinyl_chunks
        )

    with metric2:
        st.metric(
            "Caracteres extraídos",
            f"{len(pdf_bytes):,} bytes de archivo"
        )

    st.markdown("### ¿Qué quieres descubrir?")

    st.write(
        "Haz una pregunta específica o elige una de nuestras "
        "inspiraciones musicales. Las respuestas se basarán en "
        "los fragmentos más relevantes de tu documento."
    )

    question_suggestions = {
        "Resumen del álbum": (
            "Resume las ideas principales del documento y "
            "destaca sus datos más importantes."
        ),
        "Historia detrás de la música": (
            "¿Qué historia, contexto cultural y acontecimientos "
            "importantes explica el documento?"
        ),
        "Artistas y canciones": (
            "¿Qué artistas, canciones, álbumes o figuras musicales "
            "se mencionan y qué papel cumplen?"
        ),
        "Mi identidad musical": (
            "A partir de lo que describe el documento, "
            "¿qué géneros, valores o gustos musicales podría "
            "explorar una persona interesada en estos temas? "
            "Distingue los datos del documento de cualquier "
            "interpretación personal."
        ),
        "Música y emociones": (
            "¿Qué emociones, recuerdos o experiencias se relacionan "
            "con la música según el documento? Explica qué "
            "información respalda cada conclusión."
        ),
        "Cultura del vinilo": (
            "¿Qué explica el documento sobre los discos de vinilo, "
            "los formatos físicos, su historia, su fabricación "
            "o su valor cultural?"
        )
    }

    selected_prompt = st.selectbox(
        "Elige una inspiración",
        options=list(question_suggestions.keys()),
        index=0
    )

    question = st.text_area(
        "Tu pregunta",
        value=question_suggestions[selected_prompt],
        height=110,
        placeholder="Escribe aquí lo que quieres descubrir..."
    )

    if st.button("Revelar respuesta ↗", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta antes de continuar.")
        else:
            try:
                with st.spinner("Buscando las notas adecuadas..."):
                    knowledge_base = (
                        st.session_state.vinyl_knowledge_base
                    )

                    docs = knowledge_base.similarity_search(
                        question,
                        k=4
                    )

                    context = "\n\n".join(
                        f"[Fragmento {i + 1}]\n{doc.page_content}"
                        for i, doc in enumerate(docs)
                    )

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        api_key=api_key
                    )

                    messages = [
                        SystemMessage(
                            content=(
                                "Eres The Vinyl Library, un asistente "
                                "experto en análisis documental, cultura "
                                "musical e historia de los formatos "
                                "físicos. Responde en español, con un "
                                "tono cálido, editorial y claro. "
                                "Utiliza exclusivamente la información "
                                "del contexto para afirmar hechos sobre "
                                "el documento. Si falta información, "
                                "dilo con honestidad. No inventes citas, "
                                "artistas, canciones ni datos. Puedes "
                                "ofrecer interpretaciones musicales, "
                                "pero identifícalas como interpretaciones "
                                "y no como hechos del documento. "
                                "Organiza la respuesta con subtítulos "
                                "cuando ayuden a comprenderla."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"PREGUNTA:\n{question}\n\n"
                                f"CONTEXTO DEL DOCUMENTO:\n{context}\n\n"
                                "Responde basándote en estos fragmentos."
                            )
                        )
                    ]

                    response = llm.invoke(messages)

                st.markdown("---")
                st.markdown(
                    '<div class="section-label">'
                    'SIDE A — LO QUE ENCONTRAMOS'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(response.content)

                with st.expander("Consultar fragmentos utilizados"):
                    for i, doc in enumerate(docs, start=1):
                        st.markdown(f"**Fragmento {i}**")
                        st.write(doc.page_content)

                st.caption(
                    "Respuesta generada a partir de los fragmentos "
                    "recuperados del documento."
                )

            except Exception as e:
                st.error(
                    "No fue posible generar la respuesta. "
                    "Comprueba tu clave, el acceso al modelo y "
                    "tu saldo o límites de uso de la API."
                )
                st.caption(f"Detalle técnico: {str(e)}")


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.markdown("""
<div class="footer-vinyl">
    THE VINYL LIBRARY · MUSIC, MEMORY & MEANING<br>
    EVERY DOCUMENT HAS A STORY. EVERY QUESTION HAS A SOUNDTRACK.
</div>
""", unsafe_allow_html=True)
