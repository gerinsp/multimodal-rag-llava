import io
from PIL import Image
import streamlit as st

from app.config import DOCS_DIR, TOP_K_DEFAULT, LLAVA_MODE
from app.ingest import main as ingest_main
from app.rag import Retriever
from app.llava_infer import get_vlm

st.set_page_config(page_title="Multimodal RAG (LLaVA + FAISS)", layout="wide")

@st.cache_resource(show_spinner=False)
def get_retriever():
    return Retriever()

@st.cache_resource(show_spinner=False)
def get_llava():
    return get_vlm()

def sidebar_ingest():
    st.sidebar.header("📄 Kelola Dokumen")
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDF (boleh banyak)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_pdfs:
        for f in uploaded_pdfs:
            (DOCS_DIR / f.name).write_bytes(f.read())
        st.sidebar.success(f"Tersimpan {len(uploaded_pdfs)} file ke data/docs/")
    if st.sidebar.button("🔁 (Re)Build Index"):
        with st.spinner("Membangun index FAISS..."):
            ingest_main(DOCS_DIR)
        st.sidebar.success("Index selesai dibangun.")
    if st.sidebar.button("🔄 Refresh retriever"):
        get_retriever.clear()
        st.sidebar.success("Retriever di-refresh.")

def main():
    st.title("Multimodal RAG — LLaVA")
    st.info(f"Mode VLM saat ini: **{LLAVA_MODE}**")

    sidebar_ingest()

    if "history" not in st.session_state:
        st.session_state.history = []

    col1, col2 = st.columns([1, 2])
    with col1:
        image_file = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])
        img = None
        if image_file:
            img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            st.image(img, caption="Gambar input", use_container_width=True)
    with col2:
        question = st.text_area("Pertanyaan:", height=120)
        top_k = st.number_input("Top‑k dokumen", min_value=1, max_value=10, value=TOP_K_DEFAULT, step=1)
        go = st.button("Tanya ➜")

    if go:
        if not img:
            st.warning("Silakan upload gambar terlebih dahulu.")
            return
        if not question.strip():
            st.warning("Pertanyaan tidak boleh kosong.")
            return

        with st.spinner("Menjalankan retrieval..."):
            retriever = get_retriever()
            hits = retriever.search(question, k=top_k)
            contexts = [m for _, m in hits]

        with st.spinner("Menjalankan VLM..."):
            vlm = get_llava()
            answer = vlm.answer(img, question, contexts)

        st.session_state.history.append(
            {"question": question, "answer": answer, "sources": contexts}
        )

    for item in reversed(st.session_state.history):
        st.markdown("### 🧑‍💻 Pertanyaan")
        st.write(item["question"])
        st.markdown("### 🤖 Jawaban")
        st.write(item["answer"])
        st.markdown("**Sumber terpakai:**")
        for i, s in enumerate(item["sources"], 1):
            st.markdown(f"- [{i}] {s['source']} p.{s['page']}")

if __name__ == "__main__":
    main()
