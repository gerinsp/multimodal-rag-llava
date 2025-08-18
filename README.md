## Cara Menjalankan Projek RAG LLava

1. Clone repo
```bash
git clone <repo-anda> multimodal-rag-llava
```

2. Masuk ke folder projek
```bash
cd multimodal-rag-llava
```

3. Buat virtual environment

````shell
python -m venv venv && source venv/bin/activate
````   
Atau jika menggunakan windows

```shell
.venv\Scripts\activate
```
4. Install requirements

```shell
pip install -r requirements.txt
```

---
### (Opsional) Login Hugging Face jika perlu huggingface-cli login

1. Siapkan PDF

```shell
mkdir -p data/docs
```
taruh file-file PDF Anda di data/docs/

2. Bangun index (sekali atau setiap update dokumen)
```shell
python -m app.ingest --docs data/docs
```
3. Jalankan UI
```shell
streamlit run app/app.py
```
