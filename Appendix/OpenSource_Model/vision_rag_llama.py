
"""
VisionRAG: Retrieval-Augmented Generation over PDFs with Llama 3.2 11B Vision
"""

from __future__ import annotations

import base64, csv, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from io import BytesIO
from RAGModel import RAGMultiModalModel

import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image


@dataclass
class RAGResult:
    doc_id: int
    score: float
    base64: str  # base64-encoded JPEG of the page


class VisionRAG:
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        input_path: str = None,
        device: str = "cuda",
        index_root: str = ".byaldi",
        verbose: int = 1,
        llama_model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        load_in_4bit: bool = True,
        gpu_ids: str = "1",
    ):
        self.model_name = model_name
        self.device = device
        self.index_root = index_root
        self.verbose = verbose
        self.input_path = input_path
        self._rag_model: Optional[RAGMultiModalModel] = None
        self._loaded_index_name: Optional[str] = None  # cache which index is loaded

        # Llama Vision model setup
        self.llama_model_name = llama_model_name
        self.load_in_4bit = load_in_4bit
        self.gpu_ids = gpu_ids
        self._llama_model = None
        self._llama_tokenizer = None

    def _ensure_llama_loaded(self) -> None:
        """Load Llama Vision model if not already loaded."""
        if self._llama_model is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            print(f"Loading Llama 3.2 11B Vision across GPUs {self.gpu_ids}...")
            self._llama_model, self._llama_tokenizer = FastVisionModel.from_pretrained(
                self.llama_model_name,
                load_in_4bit=self.load_in_4bit,
                device_map="auto",
            )
            print("Llama Vision model loaded successfully.")

    # ---------- ColPali / byaldi I/O ----------
    def _ensure_model_loaded(self) -> None:
        if self._rag_model is None:
            self._rag_model = RAGMultiModalModel.from_pretrained(
                self.model_name,
                index_root=self.index_root,
                device=self.device,
                verbose=self.verbose,
            )

    def build_index(
        self,
        input_path: str,
        index_name: str,
        overwrite: bool = True,
        store_collection: bool = True,
    ) -> None:
        """Build an index over a single PDF or a directory of PDFs."""
        self._ensure_model_loaded()
        self._rag_model.index(
            input_path=input_path,
            index_name=index_name,
            store_collection_with_index=store_collection,
            overwrite=overwrite,
        )

    def _ensure_index_loaded(self, index_name: str) -> None:
        """Load index only if it's not already loaded in this instance."""
        if not self._index_exists(self.index_root, index_name):
            self.build_index(
                    input_path=self.input_path,
                    index_name=index_name,
                    overwrite=True,
                    store_collection=True,
                )
        if self._rag_model is None or self._loaded_index_name != index_name:
            self._rag_model = RAGMultiModalModel.from_index(
                index_path=index_name,
                index_root=self.index_root,
                device=self.device,
                verbose=self.verbose,
            )
            self._loaded_index_name = index_name

    def search(self, text_query: str, index_name: str, k: int = 3) -> List[RAGResult]:
        """Top-k search returning base64 page renders."""
        self._ensure_index_loaded(index_name)
        raw = self._rag_model.search(
            query=text_query, k=k, return_base64_results=True
        )
        return [RAGResult(doc_id=r.doc_id, score=r.score, base64=r.base64) for r in raw]

    # ---------- Helpers (CSV / files) ----------
    @staticmethod
    def _index_exists(index_root: str, index_name: str) -> bool:
        return (Path(index_root) / index_name / "index_config.json.gz").exists()

    @staticmethod
    def _read_file_as_base64(path: str) -> Tuple[str, str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"External image not found: {path}")
        ext = p.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "application/octet-stream"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return mime, b64

    @staticmethod
    def _base64_to_pil(b64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        image_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(image_data))

    @staticmethod
    def _get_question_from_csv_by_image_name(csv_path: str, image_filename: str) -> Optional[str]:
        """
        Match on an IMAGE column (case-insensitive among common variants) using the basename of the provided file.
        Return the first column's value from that row as the question.
        """
        candidates_image_cols = ["image", "img", "image_path", "image_file", "filename"]
        target_basename = Path(image_filename).name

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            # Identify the first column (question) and the image column
            headers = reader.fieldnames
            q_header = headers[0]
            lower_map = {h.lower(): h for h in headers}
            img_header = None
            for cand in candidates_image_cols:
                if cand in lower_map:
                    img_header = lower_map[cand]
                    break
            if img_header is None:
                return None

            for row in reader:
                cell = (row.get(img_header, "") or "").strip()
                if Path(cell).name == target_basename:
                    val = (row.get(q_header, "") or "").strip()
                    return val if val else None
        return None

    def answer_query(
        self,
        query: Optional[str],
        index_name: str,
        k: int = 3,
        api_key: Optional[str] = None,  # kept for compatibility, not used
        max_completion_tokens: int = 1024,
        temperature: float = 0.0,  # kept for compatibility, not used
        # External DesignQA assets
        external_image_path: Optional[str] = None,
        external_csv_path: Optional[str] = None,

        prefer_csv_prompt: bool = True,

        fallback_input_path: Optional[str] = None,
        reasoning: Optional[str] = None,  # kept for compatibility, not used
        ) -> str:
        """
        Retrieve top-k pages and answer with Llama 3.2 11B Vision.
        """
        # Ensure Llama is loaded
        self._ensure_llama_loaded()

        # 0) Ensure index exists (auto-build once if requested)
        if not self._index_exists(self.index_root, index_name):
            if fallback_input_path:
                self.build_index(
                    input_path=fallback_input_path,
                    index_name=index_name,
                    overwrite=True,
                    store_collection=True,
                )
            else:
                raise FileNotFoundError(
                    f"Missing index '.byaldi/{index_name}'. "
                    f"Run build_index(input_path=..., index_name='{index_name}') "
                    f"or pass fallback_input_path=... to auto-build."
                )

        # 1) Determine user question text
        csv_question: Optional[str] = None
        if prefer_csv_prompt and external_image_path and external_csv_path:
            csv_question = self._get_question_from_csv_by_image_name(
                external_csv_path, external_image_path
            )

        user_question = (csv_question or query or "").strip()
        if not user_question:
            raise ValueError(
                "No question available: provide `query` or a CSV with an IMAGE column and the first column as the question."
            )

        # 2) Retrieve top-k pages
        results = self.search(text_query=user_question, index_name=index_name, k=k)
        if not results:
            return "No relevant pages found for the query."

        # 3) Collect ALL images: external image first (if provided), then all k retrieved pages
        all_images = []

        # Add external image first if provided
        if external_image_path:
            all_images.append(Image.open(external_image_path))

        # Add all k retrieved page images
        for res in results:
            all_images.append(self._base64_to_pil(res.base64))

        num_retrieved = len(results)
        has_external = external_image_path is not None

        # 4) Compose message for Llama Vision
        system_prompt = (
            "You are an expert assistant specialising in analysing complex PDF pages and attached images. "
            "Carefully read the retrieved rule pages and any extra image. "
            "Ground the answer strictly in the retrieved content."
        )
        content = []

        # Add image placeholders for each image
        for _ in all_images:
            content.append({"type": "image"})

        # Add the text prompt
        content.append({"type": "text", "text": f"{system_prompt}\n\nQuestion: {user_question}"})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # 5) Process inputs for Llama Vision
        text = self._llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._llama_tokenizer(
            images=all_images,
            text=text,
            return_tensors="pt",
        ).to("cuda")

        # 6) Generate response
        with torch.no_grad():
            outputs = self._llama_model.generate(
                **inputs,
                max_new_tokens=max_completion_tokens,
                do_sample=False,
            )

        # Decode response
        response = self._llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response (Llama 3.2 format)
        if "assistant" in response.lower():
            # Try to extract after the last assistant marker
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()

        return response

if __name__ == "__main__":
    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    rag.build_index(input_path="dataset/docs/FSAE_Rules_2024_V1.pdf",
                    index_name="multimodal_rag_colqwen2.5-v0.2",
                    overwrite=True,
                    store_collection=True)
    results = rag.search("What does rule V.1.4.1 state exactly?", index_name="multimodal_rag_colqwen2.5-v0.2", k=10)
