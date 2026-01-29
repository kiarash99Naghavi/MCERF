#!/usr/bin/env python3
"""
Routing controller for DesignQA_CEaD tasks using LLM-based routing.



Two routing modes:
1. --llm-router: Full LLM routing with question + image
2. --ocr-router: OCR extraction + LLM routing with question + text only

CLI:
python Router1.py [--subtask SUBTASK] [--limit N] [--llm-router | --ocr-router]
"""

import os
import sys
import io
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from PIL import Image

# ---- Environment setup ----
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

#sys.path.append("") # Add project root to PYTHONPATH if needed

from dotenv import load_dotenv
load_dotenv()

# ---- Configuration ----
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-5-mini")
OUTPUT_DIR = Path("routing_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Available test scripts
TEST_SCRIPTS = {
    "rag": "GPT-5-MCERF-Main.py",
    "hybrid": "GPT-5-MCERF-Hybrid.py",
    "reasoning": "GPT-5-MCERF-Reasoning.py",
    "vision2text": "tests/test_gpt5_vision2text_colpali.py",
}

# Subtask configurations
SUBTASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "rule_functional_performance_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_functional_performance_qa/rule_functional_performance_qa.csv",
        "image_dir": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_functional_performance_qa/images",
        "name": "rule_functional_performance_qa",
    },
    "rule_dimension_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv",
        "image_dir": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_dimension_qa/context",
        "name": "rule_dimension_qa_context",
    },
    "rule_dimension_qa_detailed": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv",
        "image_dir": "DesignQA_CEaD-main/design_qa-main/dataset/rule_compliance/rule_dimension_qa/detailed_context",
        "name": "rule_dimension_qa_detailed_context",
    },
    "rule_definition_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_comprehension/rule_definition_qa.csv",
        "image_dir": "DesignQA_CEaD-main/design_qa-main/dataset/rule_comprehension/rule_definition_qa",
        "name": "rule_definition_qa",
    },
    "rule_presence_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_comprehension/rule_presence_qa.csv",
        "image_dir": "DesignQA_CEaD-main/design_qa-main/dataset/rule_comprehension/rule_presence_qa",
        "name": "rule_presence_qa",
    },
    "rule_compilation_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_extraction/rule_compilation_qa.csv",
        "name": "rule_compilation_qa",
    },
    "rule_retrieval_qa": {
        "csv_path": "DesignQA_CEaD-main/design_qa-main/dataset/rule_extraction/rule_retrieval_qa.csv",
        "name": "rule_retrieval_qa",
    },
}

# Sampling parameters
DEFAULT_MAX_SAMPLE = int(os.getenv("MAX_SAMPLE", "20"))
DEFAULT_SEED = int(os.getenv("SAMPLE_SEED", "42"))

# ---- Utilities ----

def _sample_df_for_voting(
    df: pd.DataFrame,
    max_sample: int = DEFAULT_MAX_SAMPLE,
    limit: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, int]:
    """Return a sampled dataframe (up to max_sample, then limit) and its length."""
    n0 = len(df)
    n_target = min(max_sample, n0) if limit is None else min(limit, max_sample, n0)
    if n0 > n_target:
        df = df.sample(n=n_target, random_state=seed).copy()
    df = df.reset_index(drop=True)
    return df, len(df)

def _encode_image_b64(path: Path) -> Tuple[str, str]:
    """Encode image to base64."""
    ext = path.suffix.lower().lstrip(".") or "png"
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("utf-8")
    return b, ext

def resolve_image_path(row: pd.Series, image_dir: Optional[str]) -> Optional[Path]:
    """Find image path for a row."""
    if not image_dir:
        return None
    for col in ("image", "img", "image_filename"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            p = Path(image_dir) / str(row[col])
            if p.exists():
                return p
    return None

# ---- OCR Text Extraction ----

def extract_text_from_image_ocr(image_path: Path) -> str:
    """Extract text from image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""

# ---- LLM Routing ----

class LLMRouter:
    """LLM-based router for test script selection."""

    def __init__(self, model: str = ROUTER_MODEL):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model
        except Exception as e:
            raise RuntimeError(f"OpenAI client initialization failed: {e}")

    def route_with_image(self, question: str, image_path: Optional[Path]) -> Dict[str, Any]:
        """Route using question and image (full LLM routing)."""
        system_prompt = """
        You are a routing system for engineering QA tasks. Choose the best test script:

        ROUTING RULES:
        1. No image: Choose RAG (simple lookup) or Hybrid (complex multi-part questions)
        2. Image with tables/charts/simulation results/text-heavy content → vision2text
        3. Image with CAD drawings/diagrams/minimal text → reasoning

        Available options:
        - "rag": For complex question requiring multiple rule finding.
        - "hybrid": For a specific rule look up that the rule name is available.
        - "reasoning": For visual analysis with minimal text content such as CAD, diagrams that has minimal text
        - "vision2text": For text-heavy technical content, tables, specifications, simulation results

        Return JSON: {"test_script": "option", "reason": "explanation"}
        """.strip()

        user_content = [{"type": "text", "text": f"Question: {question}"}]

        if image_path and image_path.exists():
            try:
                b64, ext = _encode_image_b64(image_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                    }
                )
            except Exception as e:
                print(f"Failed to encode image {image_path}: {e}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)
            script_key = data.get("test_script", "rag")
            reason = data.get("reason", "LLM routing decision")

            test_script = TEST_SCRIPTS.get(script_key, TEST_SCRIPTS["rag"])

            return {
                "test_script": test_script,
                "reason": f"LLM-Image: {reason}",
                "routing_method": "llm_with_image",
            }

        except Exception as e:
            has_image = image_path is not None and image_path.exists()
            if has_image:
                fallback_script = TEST_SCRIPTS["reasoning"]
                reason = f"LLM error, fallback to reasoning for image: {e}"
            else:
                fallback_script = TEST_SCRIPTS["rag"]
                reason = f"LLM error, fallback to RAG for text-only: {e}"

            return {
                "test_script": fallback_script,
                "reason": reason,
                "routing_method": "fallback",
            }

    def route_with_ocr(self, question: str, image_path: Optional[Path]) -> Dict[str, Any]:
        """Route using question and OCR-extracted text."""
        image_text = ""
        if image_path and image_path.exists():
            image_text = extract_text_from_image_ocr(image_path)

        system_prompt = """
        You are a routing system for engineering QA tasks. Choose the best test script:

        ROUTING RULES:
        1. No image: Choose RAG (simple lookup) or Hybrid (complex multi-part questions)
        2. Image with tables/charts/simulation results/text-heavy content → vision2text
        3. Image with CAD drawings/diagrams/minimal text → reasoning

        Available options:
        - "rag": For complex question requiring multiple rule finding.
        - "hybrid": For a specific rule look up that the rule name is available.
        - "reasoning": For visual analysis with minimal text content such as CAD, diagrams that has minimal text
        - "vision2text": For text-heavy technical content, tables, specifications, simulation results

        Return JSON: {"test_script": "option", "reason": "explanation"}
        """.strip()

        user_prompt = f"""
        Question: {question}

        Image Text: {image_text if image_text else "No image or no text in image"}
        """.strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)
            script_key = data.get("test_script", "rag")
            reason = data.get("reason", "OCR-based routing decision")

            test_script = TEST_SCRIPTS.get(script_key, TEST_SCRIPTS["rag"])

            return {
                "test_script": test_script,
                "reason": f"LLM-OCR: {reason}",
                "routing_method": "llm_with_ocr",
                "extracted_text": image_text[:200] + "..." if len(image_text) > 200 else image_text,
            }

        except Exception as e:
            has_image_text = bool(image_text.strip())
            if has_image_text:
                if len(image_text) > 100:
                    fallback_script = TEST_SCRIPTS["vision2text"]
                    reason = f"OCR fallback to vision2text (text length: {len(image_text)}): {e}"
                else:
                    fallback_script = TEST_SCRIPTS["reasoning"]
                    reason = f"OCR fallback to reasoning (text length: {len(image_text)}): {e}"
            else:
                fallback_script = TEST_SCRIPTS["rag"]
                reason = f"OCR fallback to RAG (no image text): {e}"

            return {
                "test_script": fallback_script,
                "reason": reason,
                "routing_method": "ocr_fallback",
                "extracted_text": image_text[:200] + "..." if len(image_text) > 200 else image_text,
            }

# ---- Main Processing ----

def process_subtask(
    subtask_name: str,
    config: Dict[str, Any],
    routing_mode: str = "llm",
    limit: Optional[int] = None,
    max_sample: int = DEFAULT_MAX_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> bool:
    """Process a subtask with the specified routing mode."""

    csv_path = config["csv_path"]
    image_dir = config.get("image_dir")

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return False

    if "question" not in df.columns:
        print("Error: 'question' column missing")
        return False

    router = LLMRouter()

    # Prepare output columns
    output_columns = ["router_test", "router_reason", "routing_method"]
    if routing_mode == "ocr":
        output_columns.append("extracted_text")

    for col in output_columns:
        if col not in df.columns:
            df[col] = None

    # Sample for ensemble voting
    df, n = _sample_df_for_voting(df, max_sample=max_sample, limit=limit, seed=seed)

    print(f"\nProcessing {subtask_name} with {routing_mode} routing mode...")
    print(f"Processing {n} rows...")

    for idx in range(n):
        row = df.iloc[idx]
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        img_path = resolve_image_path(row, image_dir)

        if routing_mode == "llm":
            decision = router.route_with_image(question, img_path)
        elif routing_mode == "ocr":
            decision = router.route_with_ocr(question, img_path)
        else:
            raise ValueError(f"Invalid routing mode: {routing_mode}")

        df.at[idx, "router_test"] = decision.get("test_script")
        df.at[idx, "router_reason"] = decision.get("reason")
        df.at[idx, "routing_method"] = decision.get("routing_method")

        if routing_mode == "ocr":
            df.at[idx, "extracted_text"] = decision.get("extracted_text", "")

        img_status = f"img={img_path.name}" if img_path else "no_img"
        print(
            f"Row {idx+1}/{n} -> test={Path(df.at[idx,'router_test']).name} {img_status}"
        )

    # Save individual results
    out_path = OUTPUT_DIR / f"{config.get('name', subtask_name)}_routing_{routing_mode}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Ensemble voting
    votes = df["router_test"].value_counts()
    if len(votes) > 0:
        selected_test = votes.index[0]
        support = int(votes.iloc[0])
    else:
        selected_test = TEST_SCRIPTS["rag"]
        support = 0

    support_frac = support / max(len(df), 1)

    # Save summary
    summary = {
        "subtask": subtask_name,
        "routing_mode": routing_mode,
        "n_sampled": int(len(df)),
        "vote_counts": {str(k): int(v) for k, v in votes.to_dict().items()},
        "chosen_test": str(selected_test),
        "chosen_support": support,
        "chosen_support_frac": support_frac,
    }

    summary_path = OUTPUT_DIR / f"{config.get('name', subtask_name)}_routing_summary_{routing_mode}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[Ensemble] {subtask_name} → {Path(selected_test).name} "
        f"(votes={support}/{len(df)})"
    )

    return True

def main():
    parser = argparse.ArgumentParser(description="LLM-based routing for test script selection")
    parser.add_argument("--subtask", type=str, help="Process only this subtask")
    parser.add_argument("--limit", type=int, help="Limit rows per subtask")
    parser.add_argument(
        "--llm-router",
        action="store_true",
        help="Use full LLM routing with images (default)",
    )
    parser.add_argument(
        "--ocr-router",
        action="store_true",
        help="Use OCR + LLM routing (text only)",
    )
    parser.add_argument(
        "--max-sample", type=int, default=DEFAULT_MAX_SAMPLE, help="Maximum samples for ensemble voting"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling")

    args = parser.parse_args()

    # Determine routing mode
    if args.ocr_router:
        routing_mode = "ocr"
    else:
        routing_mode = "llm"

    print(f"Using routing mode: {routing_mode}")
    print(f"Available test scripts: {list(TEST_SCRIPTS.values())}")

    ok, fail = [], []
    items = [(args.subtask, SUBTASK_CONFIGS[args.subtask])] if args.subtask else SUBTASK_CONFIGS.items()

    for subtask_name, config in items:
        try:
            success = process_subtask(
                subtask_name=subtask_name,
                config=config,
                routing_mode=routing_mode,
                limit=args.limit,
                max_sample=args.max_sample,
                seed=args.seed,
            )
            (ok if success else fail).append(subtask_name)
        except Exception as e:
            print(f"Fatal error in {subtask_name}: {e}")
            fail.append(subtask_name)

    print(f"\nSummary ({routing_mode} mode):")
    print(f"  Success: {ok}")
    print(f"  Failed: {fail}")

if __name__ == "__main__":
    main()
