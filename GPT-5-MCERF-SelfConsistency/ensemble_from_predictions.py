import os
import argparse
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI

SYSTEM_PROMPT = """You are a careful technical assistant.
You will be given up 5 candidate model outputs by running LLM 5 times to see how the answer change so we could get a final averaged answer. 

Task:
- Using ONLY the information in those candidate texts, decide the average result that makes more sense. Like if most candidates say similar thing then the avg should be that.
- Very important: Preserve the SAME format exactly. Do not add additional text or change the format. (For example, we will only have one "Explanation: " or one "Answer: " in the case that candiates also have them.)
- Do not repeat all the candidates, just their average results.

Rules:
- Do NOT reference external context or the original question.
- Do NOT invent new facts; base your decision solely on the candidates.
"""

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _key_of(row: pd.Series) -> Tuple[str, str]:
    if "question" in row and "image" in row:
        return (
            "" if pd.isna(row["question"]) else str(row["question"]),
            "" if pd.isna(row["image"]) else str(row["image"])
        )
    return (str(getattr(row, "name", "")), "")

def _build_index(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
    idx: Dict[Tuple[str, str], pd.Series] = {}
    for _, r in df.iterrows():
        idx[_key_of(r)] = r
    return idx

def _call_model(client: OpenAI, model: str, candidates: List[str]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(candidates)}
        ],
    )
    return resp.choices[0].message.content or ""

def run_ensemble(
    inputs: List[str],
    output: str,
    model: str = "gpt-5-mini",
    max_rows: Optional[int] = None,
) -> str:
    if len(inputs) != 5 and len(inputs) != 3:
        raise ValueError(f"got {len(inputs)}")

    dfs = [_load_csv(p) for p in inputs]
    idxs = [_build_index(df) for df in dfs]

    base_cols: List[str] = list(dfs[0].columns)
    if "model_prediction" not in base_cols:
        raise ValueError("First input CSV must contain 'model_prediction' column.")

    ordered_keys: List[Tuple[str, str]] = []
    seen = set()
    for df in dfs:
        for _, r in df.iterrows():
            k = _key_of(r)
            if k not in seen:
                seen.add(k)
                ordered_keys.append(k)

    if max_rows is not None:
        ordered_keys = ordered_keys[:max_rows]

    client = OpenAI()
    out_rows = {col: [] for col in base_cols}

    for k in ordered_keys:
        meta_row: Optional[pd.Series] = None
        for idx, df in zip(idxs, dfs):
            r = idx.get(k)
            if r is not None:
                meta_row = r
                break
        if meta_row is None:
            continue

        candidates = []
        for idx in idxs:
            r = idx.get(k)
            pred = ""
            if r is not None and "model_prediction" in r and not pd.isna(r["model_prediction"]):
                pred = str(r["model_prediction"])
            candidates.append(pred)

        raw = _call_model(client, model, candidates)

        for col in base_cols:
            if col == "model_prediction":
                out_rows[col].append(raw)
            else:
                out_rows[col].append(meta_row[col] if col in meta_row else "")

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    out_df.to_csv(output, index=False)
    return output

def _main_cli():
    ap = argparse.ArgumentParser(description="Ensemble final Explanation/Answer from 5 model_prediction texts.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Exactly 5 CSV input paths (order matters).")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model.")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap for quick testing.")
    args = ap.parse_args()

    out_path = run_ensemble(args.inputs, args.output, model=args.model, max_rows=args.max_rows)
    print(f"[DONE] Wrote: {out_path}")

if __name__ == "__main__":
    _main_cli()
