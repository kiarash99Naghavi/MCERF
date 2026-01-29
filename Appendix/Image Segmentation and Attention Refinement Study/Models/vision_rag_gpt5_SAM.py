"""
ColPali (byaldi) + GPT-5 Mini for PDF RAG with optional DesignQA image + CSV prompt.
Adds ROI crops support via an external directory appended AFTER the user question.
"""

from __future__ import annotations

import base64, csv, os, re, mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Replace byaldi import with your local wrapper if needed
# from byaldi import RAGMultiModalModel
from RAGModel import RAGMultiModalModel

try:
    import openai
except ImportError as exc:
    raise ImportError("vision_rag_gpt5 requires `openai`. Install it with `pip install openai`.") from exc


@dataclass
class RAGResult:
    doc_id: int
    score: float
    base64: str  # base64-encoded JPEG of the page


class VisionRAG:
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        input_path: Optional[str] = None,
        device: str = "cuda",
        index_root: str = ".byaldi",
        verbose: int = 1,
    ):
        self.model_name = model_name
        self.device = device
        self.index_root = index_root
        self.verbose = verbose
        self.input_path = input_path
        self._rag_model: Optional[RAGMultiModalModel] = None
        self._loaded_index_name: Optional[str] = None

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

    def _index_exists(self, index_root: str, index_name: str) -> bool:
        return (Path(index_root) / index_name / "index_config.json.gz").exists()

    def _ensure_index_loaded(self, index_name: str) -> None:
        if not self._index_exists(self.index_root, index_name):
            # Try building from self.input_path if set
            if self.input_path:
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
        self._ensure_index_loaded(index_name)
        raw = self._rag_model.search(
            query=text_query, k=k, return_base64_results=True
        )
        return [RAGResult(doc_id=r.doc_id, score=r.score, base64=r.base64) for r in raw]

    # ---------- Helpers (CSV / files) ----------
    @staticmethod
    def _read_file_as_base64(path: str) -> Tuple[str, str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"External image not found: {path}")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return mime, b64

    @staticmethod
    def _gather_dir_images(dir_path: str) -> List[Tuple[str, str]]:
        """Return list of (mime, base64) for images in dir, sorted by name."""
        exts = (".png", ".jpg", ".jpeg", ".webp")
        p = Path(dir_path)
        if not p.is_dir():
            return []
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        out: List[Tuple[str, str]] = []
        for f in files:
            mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            b64 = base64.b64encode(f.read_bytes()).decode("utf-8")
            out.append((mime, b64))
        return out

    @staticmethod
    def _get_question_from_csv_by_image_name(csv_path: str, image_filename: str) -> Optional[str]:
        """
        Match on an IMAGE column (case-insensitive among common variants) using the basename
        of the provided file. Return the first column's value from that row as the question.
        """
        candidates_image_cols = ["image", "img", "image_path", "image_file", "filename"]
        target_basename = Path(image_filename).name

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
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

    # ---------- GPT-5 answering ----------
    def answer_query(
        self,
        query: Optional[str],
        index_name: str,
        k: int = 3,
        api_key: Optional[str] = None,
        max_completion_tokens: int = 1024,
        temperature: float = 0.0,
        # External DesignQA assets
        external_image_path: Optional[str] = None,
        external_csv_path: Optional[str] = None,
        prefer_csv_prompt: bool = True,
        # Auto-build if index missing
        fallback_input_path: Optional[str] = None,
        reasoning: Optional[str] = None,
        # ROI directory to append AFTER the question
        external_image_dir: Optional[str] = None,
    ) -> str:
        """
        Retrieve top-k pages and answer with GPT-5.

        Behavior:
          - CSV lookup uses IMAGE column to match the provided image filename; the question is taken from the FIRST column.
          - If CSV lookup fails or not provided, falls back to `query`.
          - If the index '.byaldi/<index_name>' is missing and `fallback_input_path` is provided, the index is built.
          - Retrieved pages and optional original image are sent first, then the text question, then ROI crops.
        """
        # 0) Ensure index exists or build from fallback
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

        # 2) Retrieve relevant PDF pages
        results = self.search(text_query=user_question, index_name=index_name, k=k)
        if not results:
            return "No relevant pages found for the query."

        # 3) Compose multimodal message parts
        # Retrieved PDF pages first
        image_parts: List[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res.base64}"}}
            for res in results
        ]

        # Include the original external image with the retrieved pages (still part of image_parts)
        if external_image_path:
            mime, b64 = self._read_file_as_base64(external_image_path)
            image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        # ROI images: separate list; will be appended AFTER the text question
        roi_image_parts: List[dict] = []
        if external_image_dir:
            roi_imgs = self._gather_dir_images(external_image_dir)
            if roi_imgs:
                roi_image_parts.append({"type": "text", "text": "These are regions of interest from the original image:"})
                roi_image_parts += [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    for mime, b64 in roi_imgs
                ]

        system_prompt = (
            "You are an expert assistant specialising in analysing complex PDF pages and attached images. "
            "Carefully read the retrieved rule pages and any extra image(s). "
            "Images may include both the full original and segmented views of important parts."
            "Ground the answer strictly in the retrieved content. "
            "If retrieved information has even minor contradictions, answer 'no'. Avoid speculation."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                # Order: retrieved pages (+ original image) → text question → ROI images
                "content": image_parts + [{"type": "text", "text": user_question}] + roi_image_parts,
            },
        ]

        # key = api_key or os.getenv("OPENAI_API_KEY")
        # if not key:
        #     raise ValueError("Provide an OpenAI API key via `api_key` or OPENAI_API_KEY.")
        # client = openai.OpenAI(api_key=key)

        # resp = client.chat.completions.create(
        #     model="gpt-5-mini",
        #     reasoning_effort=reasoning,
        #     messages=messages,
        #     max_completion_tokens=max_completion_tokens,
        # )
        # return resp.choices[0].message.content
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Provide an OpenAI API key via `api_key` or OPENAI_API_KEY.")
        CLIENT = openai.OpenAI(api_key=key)

        resp = CLIENT.chat.completions.create(
            model="gpt-5-mini",
            reasoning_effort="high",
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        # return resp.choices[0].message.content, messages
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        print(f"[LLM] finish_reason={finish} usage={getattr(resp,'usage',None)} empty_first_try={not bool(content)}")

        # ── Requested fallback: resend user_question + image_parts if empty ─────
        if not content:
            fallback_msgs = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_question}] + image_parts},
            ]
            resp2 = CLIENT.chat.completions.create(
                model="gpt-5-mini",
                reasoning_effort="high",
                messages=fallback_msgs,
                max_completion_tokens=max_completion_tokens,
            )
            content = (resp2.choices[0].message.content or "No answer produced.").strip()

        return content


if __name__ == "__main__":
    # Example manual run
    rag = VisionRAG(model_name="vidore/colpali-v1.3", device="cuda")
    rag.build_index(
        input_path="dataset/docs/FSAE_Rules_2024_V1.pdf",
        index_name="multimodal_rag_colqwen2.5-v0.2",
        overwrite=True,
        store_collection=True,
    )
    _ = rag.search("What does rule V.1.4.1 state exactly?", index_name="multimodal_rag_colqwen2.5-v0.2", k=5)
