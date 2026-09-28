"""
Optional DL Refinement Layer.

For candidate pairs where the calibrated GBM probability falls in an
ambiguous band (configurable, default 0.4–0.7), run a small instruction-
tuned LLM as a binary same-business verifier.

Supported models (all license-compliant, ≤8B params):
  - Qwen/Qwen2.5-7B-Instruct   (Apache-2.0)
  - mistralai/Mistral-7B-Instruct-v0.3  (Apache-2.0)
  - microsoft/Phi-3.5-mini-instruct (MIT)

Few-shot examples are drawn from training positives/negatives —
no external data, no fine-tuning required.

This module is NOT run by default; set dl_refinement.enabled=true in config.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


_SYSTEM_PROMPT = (
    "You are a business entity matching expert. Given two business records, "
    "determine if they refer to the same real-world business entity. "
    "Respond with exactly one word: YES or NO."
)

_FEW_SHOT_TEMPLATE = """
Example {n}:
Record A: Name: {name_a} | Address: {addr_a}
Record B: Name: {name_b} | Address: {addr_b}
Same business? {label}
"""

_QUERY_TEMPLATE = """
Now decide:
Record A: Name: {name_a} | Address: {addr_a}
Record B: Name: {name_b} | Address: {addr_b}
Same business?"""


class DLRefiner:
    """
    LLM-based binary verifier for ambiguous candidate pairs.
    Uses few-shot prompting with in-context examples from training data.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        few_shot_examples: Optional[list[dict]] = None,
        max_new_tokens: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.few_shot_examples = few_shot_examples or []
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._tokenizer = None

    def load_model(self) -> "DLRefiner":
        """Lazy-load the LLM. Call only when DL refinement is enabled."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading DL refiner model: {self.model_id} on {device}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map=device,
                trust_remote_code=True,
            )
            logger.info("DL refiner model loaded.")
        except Exception as e:
            logger.error(f"Failed to load DL refiner model: {e}")
            raise
        return self

    def set_few_shot_examples(
        self,
        positive_rows: list[dict],
        negative_rows: list[dict],
        k: int = 4,
    ) -> "DLRefiner":
        """
        Build few-shot examples from training positives and negatives.
        Drawn from in-distribution data — no external examples.
        """
        examples = []
        for row in positive_rows[:k//2]:
            examples.append({
                "name_a": row.get("s1_name", ""),
                "addr_a": row.get("s1_addr", ""),
                "name_b": row.get("s23_name", ""),
                "addr_b": row.get("s23_addr", ""),
                "label": "YES",
            })
        for row in negative_rows[:k//2]:
            examples.append({
                "name_a": row.get("s1_name", ""),
                "addr_a": row.get("s1_addr", ""),
                "name_b": row.get("s23_name", ""),
                "addr_b": row.get("s23_addr", ""),
                "label": "NO",
            })
        self.few_shot_examples = examples
        return self

    def _build_prompt(self, name_a: str, addr_a: str, name_b: str, addr_b: str) -> str:
        prompt = _SYSTEM_PROMPT + "\n"
        for n, ex in enumerate(self.few_shot_examples, 1):
            prompt += _FEW_SHOT_TEMPLATE.format(n=n, **ex)
        prompt += _QUERY_TEMPLATE.format(
            name_a=name_a, addr_a=addr_a, name_b=name_b, addr_b=addr_b
        )
        return prompt

    def _predict_single(self, prompt: str) -> float:
        """Returns probability estimate: 0.9 for YES, 0.1 for NO."""
        if self._model is None or self._tokenizer is None:
            return 0.5
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        resp = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()
        if "YES" in resp:
            return 0.92
        if "NO" in resp:
            return 0.08
        return 0.5  # uncertain

    def refine_batch(
        self,
        candidates_df,
        s1_lookup: dict[str, dict],
        s23_lookup: dict[str, dict],
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Run LLM inference on the masked subset of candidate pairs.
        Returns updated probability array (same shape as mask, only masked indices filled).
        """
        probs = np.full(mask.sum(), 0.5, dtype=np.float32)
        rows = list(candidates_df.iter_rows(named=True))
        masked_rows = [r for r, m in zip(rows, mask) if m]

        for i, row in enumerate(masked_rows):
            s1 = s1_lookup.get(row["source1_id"], {})
            cand = s23_lookup.get(row["candidate_id"], {})
            prompt = self._build_prompt(
                name_a=str(s1.get("business_name", "") or ""),
                addr_a=str(s1.get("business_address", "") or ""),
                name_b=str(cand.get("business_name", "") or ""),
                addr_b=str(cand.get("business_address", "") or ""),
            )
            probs[i] = self._predict_single(prompt)
        return probs
