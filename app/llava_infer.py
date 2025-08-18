from typing import List, Dict, Protocol
import io
import base64
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from .config import (
    LLAVA_MODE, LLAVA_MODEL_NAME, LLAVA_LOCAL_PATH, HF_TOKEN,
    HF_API_URL, HF_API_TOKEN, MAX_NEW_TOKENS, TEMPERATURE
)

def build_prompt(question: str, retrieved: List[Dict]) -> str:
    ctx = "\n\n".join(
        [f"[{i+1}] {r['text']}\n(Sumber: {r['source']} p.{r['page']})" for i, r in enumerate(retrieved)]
    )
    prompt = (
        "Gunakan gambar dan konteks berikut untuk menjawab singkat, akurat, dan cantumkan nomor sumber relevan.\n"
        f"<KONTEXT>\n{ctx}\n</KONTEXT>\n\n"
        f"Pertanyaan: {question}\n"
        "Jawaban:"
    )
    return prompt

class VLM(Protocol):
    def answer(self, image: Image.Image, question: str, retrieved: List[Dict]) -> str: ...

# ========= 1) LOCAL LLaVA =========
class LocalLlava(VLM):
    def __init__(self, model_name: str = LLAVA_MODEL_NAME, local_dir: str | None = None):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        cache_dir = str(LLAVA_LOCAL_PATH if local_dir is None else local_dir)

        auth = {"token": HF_TOKEN} if HF_TOKEN else {}
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir, **auth
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            **auth
            # opsi hemat VRAM:
            # load_in_8bit=True
            # load_in_4bit=True
        )

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str, retrieved: List[Dict]) -> str:
        text = build_prompt(question, retrieved)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE
        )
        out = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        if "Jawaban:" in out:
            out = out.split("Jawaban:", 1)[-1].strip()
        return out

# ========= 2) HF INFERENCE API (LLaVA) =========
class HfApiLlava(VLM):
    def __init__(self, api_url: str = HF_API_URL, token: str = HF_API_TOKEN):
        self.api_url = api_url
        self.token = token
        if not self.api_url or not self.token:
            raise RuntimeError("HF_API_URL / HF_API_TOKEN belum diset.")

    def _img_to_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def answer(self, image: Image.Image, question: str, retrieved: List[Dict]) -> str:
        import requests
        payload = {
            "inputs": {
                "text": build_prompt(question, retrieved),
                "image": self._img_to_base64(image)  # banyak endpoint terima base64
            },
            "parameters": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE
            }
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        return str(data)

def get_vlm() -> VLM:
    if LLAVA_MODE == "local":
        return LocalLlava()
    elif LLAVA_MODE == "api":
        return HfApiLlava()
    else:
        raise ValueError(f"LLAVA_MODE tidak dikenali: {LLAVA_MODE}")
